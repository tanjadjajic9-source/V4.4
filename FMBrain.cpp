#include "FMBrain.h"
#include "WebBridge.h"

#include <winrt/Windows.ApplicationModel.h>
#include <winrt/Windows.Storage.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

// The GitHub build can also bake a copy of the compiled GPU programs into the app itself (FMShaderBlobs.h, written by build-xbox.yml).
// It is only a safety net: a program is taken from it when it cannot be read from the package's Shaders folder, so GPU training does
// not depend on exactly how the package was assembled. Builds without that header behave exactly as before.
#if __has_include("FMShaderBlobs.h")
#include "FMShaderBlobs.h"
#define FM_HAS_BUILT_IN_SHADERS 1
#endif

namespace XboxOTFM {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// ---- GPU tuning --------------------------------------------------------------------------------
constexpr uint32_t kConstantDwords = 24;                    // root constants == sizeof(GPUTrainConstants) / 4
constexpr uint32_t kReduceThreads = 4096;                   // partial sums per reduction range (64 groups x 64 threads)
constexpr uint64_t kArenaAlignFloats = 64;                  // every layer starts on a 256-byte boundary
constexpr uint64_t kActivationBudgetBytes = 224ull << 20;   // activation memory cap (the console shares memory with the app)
constexpr uint64_t kMinActivationPixels = 4ull * 32 * 32;   // first allocation: batch 4 @ 32x32
constexpr double kSubmitWorkThreshold = 1.5e9;              // ~multiply-adds recorded before a submit (keeps every submit short)
constexpr uint32_t kMaxGpuWaitMs = 15000;                   // a GPU that does not answer within 15 s is declared hung
constexpr uint32_t kMaxFailureLogLines = 40;                // never flood crash_log.txt
constexpr uint32_t kMaxDispatchDim = 65535;                 // D3D12 limit per dispatch dimension
constexpr uint32_t kLinearGroupsPerRow = 32768;             // must match kLinearRowThreads / 64 in FMCompute.hlsl

struct KernelFile {
    const wchar_t* file;
    const char* label;
};

// Order must match enum class GpuKernel in FMBrain.h.
constexpr KernelFile kKernelFiles[kGpuKernelCount] = {
    { L"CSEncodeSineOperators.cso", "CSEncodeSineOperators" },
    { L"CSConvForward.cso", "CSConvForward" },
    { L"CSMSELossGrad.cso", "CSMSELossGrad" },
    { L"CSConvBackwardInput.cso", "CSConvBackwardInput" },
    { L"CSSiLUBackward.cso", "CSSiLUBackward" },
    { L"CSConvBackwardWeights.cso", "CSConvBackwardWeights" },
    { L"CSSumSquaresPartial.cso", "CSSumSquaresPartial" },
    { L"CSReduceFinalize.cso", "CSReduceFinalize" },
    { L"CSAdamUpdate.cso", "CSAdamUpdate" },
    { L"CSEulerStep.cso", "CSEulerStep" },
};

inline float NextUniform(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return static_cast<float>((z >> 40) & 0xFFFFFFULL) / static_cast<float>(0x1000000ULL);
}

inline float NextGaussian(uint64_t& state) {
    const float u1 = std::max(1e-6f, NextUniform(state));
    const float u2 = NextUniform(state);
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * kPi * u2);
}

inline float SiLU(float x) noexcept {
    return x / (1.0f + std::exp(-std::clamp(x, -20.0f, 20.0f)));
}

void Conv2dNCHW(
    const float* input,
    const ConvLayerWeights& layer,
    uint32_t batch,
    uint32_t height,
    uint32_t width,
    float* output) {
    const uint32_t inC = layer.inChannels;
    const uint32_t outC = layer.outChannels;
    const uint32_t k = layer.kernelSize;
    const int pad = static_cast<int>(k / 2);
    const size_t hw = static_cast<size_t>(height) * width;

    for (uint32_t n = 0; n < batch; ++n) {
        const float* inBatch = input + static_cast<size_t>(n) * inC * hw;
        float* outBatch = output + static_cast<size_t>(n) * outC * hw;

        for (uint32_t oc = 0; oc < outC; ++oc) {
            const float bias = layer.bias[oc];
            float* outPlane = outBatch + static_cast<size_t>(oc) * hw;
            std::fill_n(outPlane, hw, bias);

            for (uint32_t ic = 0; ic < inC; ++ic) {
                const float* inPlane = inBatch + static_cast<size_t>(ic) * hw;
                const float* wKernel = layer.weight.data() + ((static_cast<size_t>(oc) * inC + ic) * k * k);

                if (k == 1) {
                    const float w = wKernel[0];
                    for (size_t idx = 0; idx < hw; ++idx) {
                        outPlane[idx] += inPlane[idx] * w;
                    }
                } else {
                    for (uint32_t y = 0; y < height; ++y) {
                        for (uint32_t x = 0; x < width; ++x) {
                            float acc = 0.0f;
                            for (int ky = 0; ky < static_cast<int>(k); ++ky) {
                                const int iy = static_cast<int>(y) + ky - pad;
                                if (iy < 0 || iy >= static_cast<int>(height)) continue;
                                for (int kx = 0; kx < static_cast<int>(k); ++kx) {
                                    const int ix = static_cast<int>(x) + kx - pad;
                                    if (ix < 0 || ix >= static_cast<int>(width)) continue;
                                    acc += inPlane[static_cast<size_t>(iy) * width + ix] *
                                           wKernel[static_cast<size_t>(ky) * k + kx];
                                }
                            }
                            outPlane[static_cast<size_t>(y) * width + x] += acc;
                        }
                    }
                }
            }
        }
    }
}

// ---- hung-GPU parking lot ------------------------------------------------------------------------
// If the GPU stops answering (timeout) it may still be running our last job. Freeing the buffers under it would be a
// use-after-free ON the GPU, so in that one case everything the GPU could touch is parked here (kept alive) for the rest
// of the process instead of being released. This costs some video memory once per hang, which is why GPU compute is
// switched off for the session after kMaxParkedGpuSets hangs.
constexpr uint32_t kMaxParkedGpuSets = 2;

std::mutex& ParkMutex() { static std::mutex* m = new std::mutex(); return *m; }
std::vector<IUnknown*>& ParkedObjects() { static std::vector<IUnknown*>* v = new std::vector<IUnknown*>(); return *v; }
std::vector<HANDLE>& ParkedHandles() { static std::vector<HANDLE>* v = new std::vector<HANDLE>(); return *v; }
uint32_t& ParkedSetCount() { static uint32_t n = 0; return n; }

template <class T>
void Park(Microsoft::WRL::ComPtr<T>& p) {
    if (!p) return;
    IUnknown* raw = p.Get();
    raw->AddRef();
    ParkedObjects().push_back(raw);
    p.Reset();
}

void Park(GpuBuffer& b) { Park(b.res); }

// ---- GPU helpers ---------------------------------------------------------------------------------

// Everything the GPU path wants to say goes to crash_log.txt (readable from the phone: http://<xbox>:8080/api/logs).
void GpuLog(const std::string& message) {
    try {
        WebBridge::WriteCrashLogToDisk("[GPU] " + message);
    } catch (...) {
    }
}

std::string HexHr(HRESULT hr) {
    char buf[16] = {};
    std::snprintf(buf, sizeof(buf), "0x%08X", static_cast<unsigned int>(hr));
    return std::string(buf);
}

std::string NarrowAscii(const std::wstring& w) {
    std::string s;
    s.reserve(w.size());
    for (wchar_t c : w) {
        s.push_back((c >= 32 && c < 127) ? static_cast<char>(c) : '?');
    }
    return s;
}

inline uint64_t AlignUp(uint64_t value, uint64_t alignment) noexcept {
    return (value + alignment - 1) / alignment * alignment;
}

bool ReadWholeFile(const std::wstring& path, std::vector<uint8_t>& outBytes) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) return false;
    const std::streamoff size = static_cast<std::streamoff>(ifs.tellg());
    if (size <= 0) return false;
    outBytes.resize(static_cast<size_t>(size));
    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(outBytes.data()), static_cast<std::streamsize>(size));
    return !ifs.fail();
}

// Looks for Shaders\<filename> inside the installed app package (normal case), then relative to the working folder.
bool LoadShaderBytecode(const wchar_t* filename, std::vector<uint8_t>& outBytes, std::wstring& outTriedPath) {
    outBytes.clear();
    outTriedPath.clear();
    try {
        const auto installPath = winrt::Windows::ApplicationModel::Package::Current().InstalledLocation().Path();
        outTriedPath = std::wstring(installPath.c_str()) + L"\\Shaders\\" + filename;
        if (ReadWholeFile(outTriedPath, outBytes)) return true;
    } catch (...) {
    }
    const std::wstring relative = std::wstring(L"Shaders\\") + filename;
    if (outTriedPath.empty()) outTriedPath = relative;
    return ReadWholeFile(relative, outBytes);
}

// Copies the copy of a compiled GPU program that was baked into the app at build time (if this build has one).
bool CopyBuiltInShader(const char* label, std::vector<uint8_t>& outBytes) {
    outBytes.clear();
#ifdef FM_HAS_BUILT_IN_SHADERS
    for (const auto& blob : EmbeddedShaders::kBlobs) {
        if (std::strcmp(blob.name, label) == 0 && blob.size >= 4) {
            outBytes.assign(blob.data, blob.data + blob.size);
            return true;
        }
    }
#else
    (void)label;
#endif
    return false;
}

// Where each conv layer's weights / bias live inside the flat parameter arenas.
void BuildLayerLayout(const BrainArchitecture& arch, std::vector<GpuLayerInfo>& layers, uint64_t& totalFloats) {
    layers.clear();
    uint64_t cursor = 0;
    auto add = [&](uint32_t inC, uint32_t outC, uint32_t k) {
        GpuLayerInfo li;
        li.inC = inC;
        li.outC = outC;
        li.k = k;
        li.wCount = static_cast<uint64_t>(outC) * inC * k * k;
        li.bCount = outC;
        li.wOff = cursor;
        cursor = AlignUp(cursor + li.wCount, kArenaAlignFloats);
        li.bOff = cursor;
        cursor = AlignUp(cursor + li.bCount, kArenaAlignFloats);
        layers.push_back(li);
    };
    add(arch.TotalInputChannels(), arch.width, 1);
    for (uint32_t b = 0; b < arch.numResBlocks; ++b) {
        add(arch.width, arch.width, 3);
        add(arch.width, arch.width, 3);
    }
    add(arch.width, arch.inputImageChannels, 1);
    totalFloats = cursor;
}

// Pointers to every CPU-side array of every layer, in the same order as BuildLayerLayout.
struct LayerVecs {
    ConvLayerWeights* layer;
    std::vector<float>* mW;
    std::vector<float>* vW;
    std::vector<float>* mB;
    std::vector<float>* vB;
};

std::vector<LayerVecs> CollectLayers(FMBrain& brain) {
    std::vector<LayerVecs> v;
    v.push_back({ &brain.Stem(), &brain.StemMW(), &brain.StemVW(), &brain.StemMB(), &brain.StemVB() });
    for (auto& blk : brain.Blocks()) {
        v.push_back({ &blk.conv1, &blk.mW1, &blk.vW1, &blk.mB1, &blk.vB1 });
        v.push_back({ &blk.conv2, &blk.mW2, &blk.vW2, &blk.mB2, &blk.vB2 });
    }
    v.push_back({ &brain.Head(), &brain.HeadMW(), &brain.HeadVW(), &brain.HeadMB(), &brain.HeadVB() });
    return v;
}

GPUTrainConstants BaseConstants(uint32_t batch, uint32_t height, uint32_t width, uint32_t imageChannels) {
    GPUTrainConstants cb = {};
    cb.gBatch = batch;
    cb.gWidth = width;
    cb.gHeight = height;
    cb.gSampleStride = imageChannels * height * width;
    return cb;
}

// 8x8 thread tiles over the image, depth = number of (sample, channel) planes.
void SetImageGrid(GpuDispatch& d, uint32_t width, uint32_t height, uint32_t planes) {
    d.gx = (width + 7) / 8;
    d.gy = (height + 7) / 8;
    d.gz = planes;
}

// 64 threads per group, folded into 2-D so even huge tensors stay under the 65535-groups-per-dimension limit.
void SetLinearGrid(GpuDispatch& d, uint64_t count) {
    const uint64_t groups = std::max<uint64_t>(1, (count + 63) / 64);
    d.gx = static_cast<uint32_t>(std::min<uint64_t>(groups, kLinearGroupsPerRow));
    d.gy = static_cast<uint32_t>((groups + kLinearGroupsPerRow - 1) / kLinearGroupsPerRow);
    d.gz = 1;
}

} // namespace

// =================================================================================================
// Model definition / CPU side
// =================================================================================================

FMBrain::FMBrain(BrainPreset preset, uint32_t waveLevel) {
    InitializeFromScratch(preset, 0x1337BEEFULL, waveLevel);
}

FMBrain::~FMBrain() {
    try {
        std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
        ReleaseGPULocked();
    } catch (...) {
    }
}

BrainArchitecture FMBrain::PresetConfig(BrainPreset preset, uint32_t waveLevel) {
    BrainArchitecture arch;
    arch.preset = preset;
    arch.waveLevel = std::min(waveLevel, kMaxWaveLevel);
    arch.inputImageChannels = ImageChannelsForWaveLevel(arch.waveLevel);
    arch.coordChannels = 2;
    arch.harmonicBands = 4;
    arch.timeChannels = 4;

    switch (preset) {
        case BrainPreset::Low25k:
            arch.width = 26;
            arch.numResBlocks = 2;
            break;
        case BrainPreset::Medium80k:
            arch.width = 38;
            arch.numResBlocks = 3;
            break;
        case BrainPreset::Max250k:
            arch.width = 58;
            arch.numResBlocks = 4;
            break;
    }
    return arch;
}

uint64_t FMBrain::ExpectedParameterCount(BrainPreset preset, uint32_t waveLevel) noexcept {
    const BrainArchitecture arch = PresetConfig(preset, waveLevel);
    const uint64_t cIn = arch.TotalInputChannels();
    const uint64_t w = arch.width;
    const uint64_t stem = (w * cIn * 1ULL * 1ULL) + w;
    const uint64_t blk = 2ULL * ((w * w * 3ULL * 3ULL) + w);
    const uint64_t head = (arch.inputImageChannels * w * 1ULL * 1ULL) + arch.inputImageChannels;
    return stem + static_cast<uint64_t>(arch.numResBlocks) * blk + head;
}

void FMBrain::InitConvLayer(ConvLayerWeights& layer, uint32_t inC, uint32_t outC, uint32_t k, uint64_t& rngState) {
    layer.inChannels = inC;
    layer.outChannels = outC;
    layer.kernelSize = k;
    const size_t weightCount = static_cast<size_t>(outC) * inC * k * k;
    layer.weight.resize(weightCount);
    layer.bias.assign(outC, 0.0f);

    const float fanIn = static_cast<float>(inC * k * k);
    const float stdDev = std::sqrt(2.0f / std::max(1.0f, fanIn));
    for (size_t i = 0; i < weightCount; ++i) {
        layer.weight[i] = NextGaussian(rngState) * stdDev;
    }
}

void FMBrain::InitializeFromScratch(BrainPreset preset, uint64_t seed, uint32_t waveLevel) {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    InitializeFromScratchLocked(preset, seed, waveLevel);
}

void FMBrain::InitializeFromScratchLocked(BrainPreset preset, uint64_t seed, uint32_t waveLevel) {
    m_arch = PresetConfig(preset, waveLevel);
    uint64_t rngState = seed;

    const uint32_t cIn = m_arch.TotalInputChannels();
    const uint32_t w = m_arch.width;

    InitConvLayer(m_stem, cIn, w, 1, rngState);
    m_stemMW.assign(m_stem.weight.size(), 0.0f);
    m_stemVW.assign(m_stem.weight.size(), 0.0f);
    m_stemMB.assign(m_stem.bias.size(), 0.0f);
    m_stemVB.assign(m_stem.bias.size(), 0.0f);

    m_blocks.resize(m_arch.numResBlocks);
    for (auto& blk : m_blocks) {
        InitConvLayer(blk.conv1, w, w, 3, rngState);
        InitConvLayer(blk.conv2, w, w, 3, rngState);
        blk.mW1.assign(blk.conv1.weight.size(), 0.0f);
        blk.vW1.assign(blk.conv1.weight.size(), 0.0f);
        blk.mB1.assign(blk.conv1.bias.size(), 0.0f);
        blk.vB1.assign(blk.conv1.bias.size(), 0.0f);

        blk.mW2.assign(blk.conv2.weight.size(), 0.0f);
        blk.vW2.assign(blk.conv2.weight.size(), 0.0f);
        blk.mB2.assign(blk.conv2.bias.size(), 0.0f);
        blk.vB2.assign(blk.conv2.bias.size(), 0.0f);
    }

    InitConvLayer(m_head, w, m_arch.inputImageChannels, 1, rngState);
    for (float& val : m_head.weight) {
        val *= 0.1f;
    }
    m_headMW.assign(m_head.weight.size(), 0.0f);
    m_headVW.assign(m_head.weight.size(), 0.0f);
    m_headMB.assign(m_head.bias.size(), 0.0f);
    m_headVB.assign(m_head.bias.size(), 0.0f);

    // The CPU vectors are the truth now; any GPU copy is stale until the next upload.
    m_gpuWeightsDirtyOnDevice = false;
    m_gpuNeedsUpload = true;
}

uint64_t FMBrain::ParameterCount() const noexcept {
    uint64_t total = m_stem.weight.size() + m_stem.bias.size();
    for (const auto& blk : m_blocks) {
        total += blk.conv1.weight.size() + blk.conv1.bias.size();
        total += blk.conv2.weight.size() + blk.conv2.bias.size();
    }
    total += m_head.weight.size() + m_head.bias.size();
    return total;
}

void FMBrain::NotifyCpuWeightsChanged() {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    m_gpuNeedsUpload = true;
    m_gpuWeightsDirtyOnDevice = false;
}

// =================================================================================================
// GPU: buffers and layout
// =================================================================================================

HRESULT FMBrain::CreateGpuBuffer(uint64_t byteSize, D3D12_HEAP_TYPE heapType, GpuBuffer& out) {
    out = GpuBuffer();
    if (!m_d3dDevice) return E_NOT_VALID_STATE;

    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = heapType;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = std::max<uint64_t>(256, AlignUp(byteSize, 256));
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    D3D12_RESOURCE_STATES initial = D3D12_RESOURCE_STATE_COMMON;
    bool fixedState = false;
    if (heapType == D3D12_HEAP_TYPE_DEFAULT) {
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    } else if (heapType == D3D12_HEAP_TYPE_UPLOAD) {
        initial = D3D12_RESOURCE_STATE_GENERIC_READ;
        fixedState = true;
    } else {
        initial = D3D12_RESOURCE_STATE_COPY_DEST;
        fixedState = true;
    }

    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    const HRESULT hr = m_d3dDevice->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &desc, initial, nullptr, IID_PPV_ARGS(&resource));
    if (FAILED(hr)) return hr;

    out.res = resource;
    out.state = initial;
    out.bytes = desc.Width;
    out.fixedState = fixedState;
    out.uavDirty = false;
    return S_OK;
}

uint64_t FMBrain::ActivationBytesPerPixel() const noexcept {
    // hidden-sized tensors per pixel: act (B+1) + midPre (B) + midAct (B) + 3 gradient buffers + stemPre
    const uint64_t hiddenTensors = 3ull * m_arch.numResBlocks + 5ull;
    // plus the encoded input (25 or 34 channels) and xt / vTarget / vPred / gradPred (3 or 12 channels each)
    return 4ull * (hiddenTensors * m_arch.width + m_arch.TotalInputChannels() + 4ull * m_arch.inputImageChannels);
}

uint32_t FMBrain::MaxGpuBatchForResolution(uint32_t resolution) const noexcept {
    const uint64_t res = std::max<uint32_t>(1u, resolution);
    const uint64_t perSample = std::max<uint64_t>(1, res * res * ActivationBytesPerPixel());
    const uint64_t fit = kActivationBudgetBytes / perSample;
    uint32_t cap = static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(fit, 1), 32));
    if (resolution >= 128u) {
        cap = std::min(cap, 2u); // 128x128 stays at a tiny batch so a single submit can never run long
    }
    return cap;
}

HRESULT FMBrain::EnsureActivationBuffersLocked(uint64_t neededPixels) {
    if (m_act.xt.res && neededPixels <= m_act.capacityPixels) return S_OK;
    if (m_arenaBytes == 0 || !m_d3dDevice) return E_NOT_VALID_STATE;

    HRESULT hr = S_OK;
    if (m_inFlight) {
        hr = WaitComputeQueue();
        if (FAILED(hr)) return hr;
    }

    const uint64_t pixels = AlignUp(std::max<uint64_t>({ neededPixels, m_act.capacityPixels, kMinActivationPixels }), 1024);
    const uint32_t width = m_arch.width;
    const uint32_t blocks = m_arch.numResBlocks;
    const uint32_t cIn = m_arch.TotalInputChannels();
    const uint64_t rgbBytes = pixels * m_arch.inputImageChannels * sizeof(float); // (3 colours, or 12 wave channels)
    const uint64_t encBytes = pixels * cIn * sizeof(float);
    const uint64_t hidBytes = pixels * width * sizeof(float);

    // Free the old set first so the old and new sets never exist at the same time.
    m_act = GpuActivations();

    auto make = [&](uint64_t bytes, GpuBuffer& b) { return CreateGpuBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT, b); };
    GpuActivations next;
    next.act.resize(blocks + 1);
    next.midPre.resize(blocks);
    next.midAct.resize(blocks);

    bool ok = true;
    ok = ok && SUCCEEDED(hr = make(rgbBytes, next.xt));
    ok = ok && SUCCEEDED(hr = make(rgbBytes, next.vTarget));
    ok = ok && SUCCEEDED(hr = make(64 * sizeof(float), next.tValues));
    ok = ok && SUCCEEDED(hr = make(encBytes, next.stemIn));
    ok = ok && SUCCEEDED(hr = make(hidBytes, next.stemPre));
    ok = ok && SUCCEEDED(hr = make(rgbBytes, next.vPred));
    ok = ok && SUCCEEDED(hr = make(rgbBytes, next.gradPred));
    ok = ok && SUCCEEDED(hr = make(hidBytes, next.gradA));
    ok = ok && SUCCEEDED(hr = make(hidBytes, next.gradB));
    ok = ok && SUCCEEDED(hr = make(hidBytes, next.gradTmp));
    for (uint32_t b = 0; ok && b <= blocks; ++b) {
        ok = ok && SUCCEEDED(hr = make(hidBytes, next.act[b]));
    }
    for (uint32_t b = 0; ok && b < blocks; ++b) {
        ok = ok && SUCCEEDED(hr = make(hidBytes, next.midPre[b]));
        ok = ok && SUCCEEDED(hr = make(hidBytes, next.midAct[b]));
    }
    if (ok) {
        const uint64_t uploadBytes = std::max<uint64_t>(2 * rgbBytes + 1024, 3 * m_arenaBytes);
        const uint64_t readbackBytes = std::max<uint64_t>({ 3 * m_arenaBytes, rgbBytes, 4096ull });
        ok = ok && SUCCEEDED(hr = CreateGpuBuffer(uploadBytes, D3D12_HEAP_TYPE_UPLOAD, next.upload));
        ok = ok && SUCCEEDED(hr = CreateGpuBuffer(readbackBytes, D3D12_HEAP_TYPE_READBACK, next.readback));
    }
    if (!ok) {
        GpuLog("could not allocate activation buffers for " + std::to_string(pixels) + " pixels (HRESULT " + HexHr(hr) + ")");
        return FAILED(hr) ? hr : E_FAIL;
    }

    next.capacityPixels = pixels;
    m_act = std::move(next);
    return S_OK;
}

bool FMBrain::GpuLayoutMatchesLocked() const noexcept {
    return m_gpuLayoutValid &&
           m_gpuArch.preset == m_arch.preset &&
           m_gpuArch.width == m_arch.width &&
           m_gpuArch.numResBlocks == m_arch.numResBlocks &&
           m_gpuArch.waveLevel == m_arch.waveLevel &&
           m_gpuArch.inputImageChannels == m_arch.inputImageChannels;
}

HRESULT FMBrain::RebuildGpuLayoutLocked() {
    if (!m_d3dDevice || !m_computeQueue) return E_NOT_VALID_STATE;

    HRESULT hr = S_OK;
    if (m_inFlight) {
        hr = WaitComputeQueue();
        if (FAILED(hr)) return hr;
    }

    m_gpuLayoutValid = false;
    m_act = GpuActivations();
    m_gpuParams = GpuBuffer();
    m_gpuGrads = GpuBuffer();
    m_gpuAdamM = GpuBuffer();
    m_gpuAdamV = GpuBuffer();
    m_gpuPartials = GpuBuffer();
    m_gpuStats = GpuBuffer();

    BuildLayerLayout(m_arch, m_gpuLayers, m_arenaFloats);
    m_arenaBytes = m_arenaFloats * sizeof(float);

    if (FAILED(hr = CreateGpuBuffer(m_arenaBytes, D3D12_HEAP_TYPE_DEFAULT, m_gpuParams))) return hr;
    if (FAILED(hr = CreateGpuBuffer(m_arenaBytes, D3D12_HEAP_TYPE_DEFAULT, m_gpuGrads))) return hr;
    if (FAILED(hr = CreateGpuBuffer(m_arenaBytes, D3D12_HEAP_TYPE_DEFAULT, m_gpuAdamM))) return hr;
    if (FAILED(hr = CreateGpuBuffer(m_arenaBytes, D3D12_HEAP_TYPE_DEFAULT, m_gpuAdamV))) return hr;
    if (FAILED(hr = CreateGpuBuffer(2ull * kReduceThreads * sizeof(float), D3D12_HEAP_TYPE_DEFAULT, m_gpuPartials))) return hr;
    if (FAILED(hr = CreateGpuBuffer(16 * sizeof(float), D3D12_HEAP_TYPE_DEFAULT, m_gpuStats))) return hr;
    if (FAILED(hr = EnsureActivationBuffersLocked(kMinActivationPixels))) return hr;
    if (FAILED(hr = ZeroGradArenaLocked())) return hr;

    m_gpuArch = m_arch;
    m_gpuLayoutValid = true;
    m_gpuNeedsUpload = true;
    return S_OK;
}

HRESULT FMBrain::ZeroGradArenaLocked() {
    // Padding between layers is never written by a kernel, so it must start at exactly zero (the Adam kernel
    // sweeps the whole arena and relies on zero gradients there to leave the padding untouched).
    HRESULT hr = GpuBegin();
    if (FAILED(hr)) return hr;

    uint8_t* mapped = nullptr;
    hr = m_act.upload.res->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    if (FAILED(hr)) {
        GpuAbort();
        return hr;
    }
    std::memset(mapped, 0, static_cast<size_t>(m_arenaBytes));
    m_act.upload.res->Unmap(0, nullptr);

    RecordUploadCopy(m_gpuGrads, 0, 0, m_arenaBytes);
    return GpuFinishAndWait();
}

HRESULT FMBrain::EnsureGpuInSyncLocked() {
    if (!GpuLayoutMatchesLocked() || m_gpuNeedsUpload) {
        const HRESULT hr = UploadWeightsToGPULocked();
        if (FAILED(hr)) return hr;
    }
    // The first time a network layout (pixel or wave brain) is used on this GPU, prove its maths against the CPU exactly as at
    // start-up. Only a layout that was not tested yet gets here, right after its buffers were built and the weights uploaded.
    const uint32_t bit = 1u << m_arch.waveLevel;
    if (m_gpuReady && (m_verifiedLayouts & bit) == 0) {
        m_verifiedLayouts |= bit;
        std::string report;
        const HRESULT hr = SelfTestGpuLocked(report);
        GpuLog(std::string(m_arch.waveLevel ? "wave brain: " : "pixel brain: ") + report);
        if (FAILED(hr)) {
            m_verifiedLayouts &= ~bit;
            GpuLog("the GPU maths for this brain type did not pass the check: this brain trains on the processor instead.");
            return hr;
        }
    }
    return S_OK;
}

// =================================================================================================
// GPU: initialisation / release
// =================================================================================================

HRESULT FMBrain::InitializeGPU(ID3D12Device* d3dDevice, ID3D12CommandQueue* cmdQueue) {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    ReleaseGPULocked();

    if (!d3dDevice || !cmdQueue) {
        GpuLog("InitializeGPU was called without a D3D12 device or command queue.");
        return E_INVALIDARG;
    }
    {
        std::lock_guard<std::mutex> parkLock(ParkMutex());
        if (ParkedSetCount() >= kMaxParkedGpuSets) {
            GpuLog("GPU compute is switched off for this session: the GPU stopped answering " +
                   std::to_string(ParkedSetCount()) + " times. Restart the app to try again. Training continues on the CPU.");
            return E_FAIL;
        }
    }

    GpuLog("---- GPU compute initialisation: preset " + std::to_string(static_cast<uint32_t>(m_arch.preset)) +
           (m_arch.waveLevel ? " (wave brain)" : "") + ", " + std::to_string(ParameterCount()) + " parameters ----");

    auto fail = [&](const std::string& what, HRESULT hr) -> HRESULT {
        GpuLog("FAILED: " + what + " (HRESULT " + HexHr(hr) + "). Training continues on the CPU.");
        ReleaseGPULocked();
        return FAILED(hr) ? hr : E_FAIL;
    };

    // The shaders hard-code the sine-operator layout around the image channels: 2 coordinates, 4 harmonic bands, 4 time channels.
    if ((m_arch.inputImageChannels != 3 && m_arch.inputImageChannels != 12) || m_arch.coordChannels != 2 || m_arch.harmonicBands != 4 ||
        m_arch.timeChannels != 4 || m_arch.TotalInputChannels() != m_arch.inputImageChannels + 22) {
        return fail("this network layout is not supported by the compute shaders", E_NOT_VALID_STATE);
    }

    m_d3dDevice = d3dDevice;
    m_cmdQueue = cmdQueue;
    m_gpuFaulted = false;
    m_failureLogCount = 0;
    m_loggedFirstStep = false;
    m_loggedBatchClampRes = 0;
    m_verifiedLayouts = 0;

    // Heavy compute must not share the queue that presents the TV picture, or the UI would freeze behind it.
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    HRESULT hr = d3dDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_computeQueue));
    if (SUCCEEDED(hr)) {
        m_listType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        GpuLog("using a dedicated compute queue");
    } else {
        m_computeQueue = m_cmdQueue;
        m_listType = D3D12_COMMAND_LIST_TYPE_DIRECT;
        GpuLog("dedicated compute queue unavailable (HRESULT " + HexHr(hr) + "); sharing the graphics queue");
    }

    hr = d3dDevice->CreateCommandAllocator(m_listType, IID_PPV_ARGS(&m_computeAlloc));
    if (FAILED(hr)) return fail("CreateCommandAllocator", hr);
    hr = d3dDevice->CreateCommandList(0, m_listType, m_computeAlloc.Get(), nullptr, IID_PPV_ARGS(&m_computeCmdList));
    if (FAILED(hr)) return fail("CreateCommandList", hr);
    m_computeCmdList->Close();

    hr = d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_computeFence));
    if (FAILED(hr)) return fail("CreateFence", hr);
    m_computeFenceValue = 0;
    m_computeFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!m_computeFenceEvent) return fail("CreateEvent", E_FAIL);

    // Root signature: [0] 24 x 32-bit constants (cbuffer b0), [1..4] four root SRVs (t0..t3), [5..7] three root UAVs (u0..u2)
    D3D12_ROOT_PARAMETER params[8] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.Num32BitValues = kConstantDwords;
    for (UINT i = 0; i < 4; ++i) {
        params[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[1 + i].Descriptor.ShaderRegister = i;
    }
    for (UINT i = 0; i < 3; ++i) {
        params[5 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[5 + i].Descriptor.ShaderRegister = i;
    }
    D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
    rsDesc.NumParameters = 8;
    rsDesc.pParameters = params;
    Microsoft::WRL::ComPtr<ID3DBlob> sigBlob, errBlob;
    hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob);
    if (FAILED(hr)) return fail("D3D12SerializeRootSignature", hr);
    hr = d3dDevice->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&m_rootSig));
    if (FAILED(hr)) return fail("CreateRootSignature", hr);

    // One pipeline per kernel. Every problem is collected so ONE log shows everything that is wrong.
    std::string problems;
    std::string builtInNotes;
    uint32_t builtInCount = 0;
    for (uint32_t k = 0; k < kGpuKernelCount; ++k) {
        std::vector<uint8_t> bytecode;
        std::wstring triedPath;
        const bool fileRead = LoadShaderBytecode(kKernelFiles[k].file, bytecode, triedPath);
        const bool fileValid = fileRead && bytecode.size() >= 4 && std::memcmp(bytecode.data(), "DXBC", 4) == 0;
        if (!fileValid) {
            // Not usable from the Shaders folder: try the copy that is built into the app.
            std::vector<uint8_t> builtIn;
            if (CopyBuiltInShader(kKernelFiles[k].label, builtIn)) {
                builtInNotes += std::string("  ") + kKernelFiles[k].label +
                                (fileRead ? ": the file is not a compiled shader: " : ": file not found: ") + NarrowAscii(triedPath) + "\n";
                bytecode.swap(builtIn);
                ++builtInCount;
            } else if (!fileRead) {
                problems += std::string("  missing shader file: ") + NarrowAscii(triedPath) + "\n";
                continue;
            } else {
                problems += std::string("  not a compiled shader (no DXBC header): ") + NarrowAscii(triedPath) + "\n";
                continue;
            }
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_rootSig.Get();
        psoDesc.CS.pShaderBytecode = bytecode.data();
        psoDesc.CS.BytecodeLength = bytecode.size();
        const HRESULT psoHr = d3dDevice->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_pso[k]));
        if (FAILED(psoHr)) {
            problems += std::string("  CreateComputePipelineState failed for ") + kKernelFiles[k].label +
                        " (HRESULT " + HexHr(psoHr) + ")\n";
        }
    }
    if (!problems.empty()) {
        GpuLog("shader problems:\n" + problems);
        return fail("could not load all compute shaders", HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND));
    }
    if (builtInCount > 0) {
        GpuLog("these GPU programs were not usable from the package's Shaders folder, so the copy built into the app is used instead:\n" + builtInNotes);
    }
    GpuLog("all " + std::to_string(kGpuKernelCount) + " compute shaders loaded (" + std::to_string(kGpuKernelCount - builtInCount) +
           " from the Shaders folder, " + std::to_string(builtInCount) + " from the copy built into the app)");

    hr = RebuildGpuLayoutLocked();
    if (FAILED(hr)) return fail("allocating GPU buffers", hr);
    hr = UploadWeightsToGPULocked();
    if (FAILED(hr)) return fail("uploading the weights", hr);

    // Prove the GPU math against the CPU path before the app is allowed to claim GPU training.
    std::string report;
    hr = SelfTestGpuLocked(report);
    GpuLog(report);
    if (FAILED(hr)) return fail("GPU self-test", hr);
    m_verifiedLayouts = 1u << m_arch.waveLevel;

    // DirectML probe (informational only; the training math runs on the compute shaders above).
    try {
        if (SUCCEEDED(DMLCreateDevice(d3dDevice, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&m_dmlDevice)))) {
            if (SUCCEEDED(m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_dmlRecorder)))) {
                m_dmlReady = true;
            }
        }
    } catch (...) {
        m_dmlDevice.Reset();
        m_dmlRecorder.Reset();
        m_dmlReady = false;
    }
    if (!m_dmlReady) {
        m_dmlDevice.Reset();
        m_dmlRecorder.Reset();
    }

    m_lastWeightSync = std::chrono::steady_clock::now();
    m_gpuReady = true;
    GpuLog("GPU compute READY: " + std::to_string(ParameterCount()) + " parameters, parameter arena " +
           std::to_string(m_arenaBytes / 1024) + " KB, activation capacity " +
           std::to_string(m_act.capacityPixels) + " pixels, DirectML probe " + (m_dmlReady ? "yes" : "no"));
    return S_OK;
}

void FMBrain::ReleaseGPU() {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    ReleaseGPULocked();
}

void FMBrain::ParkGpuObjectsLocked() {
    std::lock_guard<std::mutex> lock(ParkMutex());
    ++ParkedSetCount();
    GpuLog("the GPU never finished its last job: its buffers are kept alive instead of being freed under it (hang " +
           std::to_string(ParkedSetCount()) + " of " + std::to_string(kMaxParkedGpuSets) + " allowed this session)");
    Park(m_act.xt);
    Park(m_act.vTarget);
    Park(m_act.tValues);
    Park(m_act.stemIn);
    Park(m_act.stemPre);
    Park(m_act.vPred);
    Park(m_act.gradPred);
    Park(m_act.gradA);
    Park(m_act.gradB);
    Park(m_act.gradTmp);
    for (GpuBuffer& b : m_act.act) Park(b);
    for (GpuBuffer& b : m_act.midPre) Park(b);
    for (GpuBuffer& b : m_act.midAct) Park(b);
    Park(m_act.upload);
    Park(m_act.readback);
    Park(m_gpuParams);
    Park(m_gpuGrads);
    Park(m_gpuAdamM);
    Park(m_gpuAdamV);
    Park(m_gpuPartials);
    Park(m_gpuStats);
    for (auto& pso : m_pso) Park(pso);
    Park(m_rootSig);
    Park(m_computeCmdList);
    Park(m_computeAlloc);
    Park(m_computeFence);
    if (m_computeQueue.Get() != m_cmdQueue.Get()) Park(m_computeQueue); // the graphics queue belongs to the app
    if (m_computeFenceEvent) {
        ParkedHandles().push_back(m_computeFenceEvent); // a fence may still signal it later
        m_computeFenceEvent = nullptr;
    }
}

void FMBrain::ReleaseParkedGpuObjectsForTests() {
    std::lock_guard<std::mutex> lock(ParkMutex());
    for (IUnknown* p : ParkedObjects()) p->Release();
    ParkedObjects().clear();
    for (HANDLE h : ParkedHandles()) CloseHandle(h);
    ParkedHandles().clear();
    ParkedSetCount() = 0;
}

void FMBrain::ReleaseGPULocked() {
    // A GPU that stopped answering (but was not removed) may still be running our last job: park, do not free.
    if (m_inFlight && m_gpuFaulted && m_d3dDevice && SUCCEEDED(m_d3dDevice->GetDeviceRemovedReason())) {
        ParkGpuObjectsLocked();
    }
    // Let in-flight GPU work finish before freeing what it uses (skipped if the GPU is already known to be dead).
    if (m_inFlight && !m_gpuFaulted && m_computeQueue && m_computeFence && m_computeFenceEvent) {
        WaitComputeQueue();
    }
    if (m_recording && m_computeCmdList) {
        m_computeCmdList->Close();
    }

    m_gpuReady = false;
    m_dmlReady = false;
    m_gpuLayoutValid = false;
    m_recording = false;
    m_inFlight = false;
    m_pendingWork = 0.0;
    m_verifiedLayouts = 0;

    m_act = GpuActivations();
    m_gpuParams = GpuBuffer();
    m_gpuGrads = GpuBuffer();
    m_gpuAdamM = GpuBuffer();
    m_gpuAdamV = GpuBuffer();
    m_gpuPartials = GpuBuffer();
    m_gpuStats = GpuBuffer();
    m_gpuLayers.clear();
    m_arenaFloats = 0;
    m_arenaBytes = 0;

    for (auto& pso : m_pso) pso.Reset();
    m_rootSig.Reset();

    m_dmlRecorder.Reset();
    m_dmlDevice.Reset();

    if (m_computeFenceEvent) {
        CloseHandle(m_computeFenceEvent);
        m_computeFenceEvent = nullptr;
    }
    m_computeFence.Reset();
    m_computeCmdList.Reset();
    m_computeAlloc.Reset();
    m_computeQueue.Reset();
    m_cmdQueue.Reset();
    m_d3dDevice.Reset();
}

// =================================================================================================
// GPU: queue + command recording
// =================================================================================================

HRESULT FMBrain::WaitComputeQueue() {
    if (!m_computeQueue || !m_computeFence || !m_computeFenceEvent || !m_d3dDevice) return E_NOT_VALID_STATE;

    const uint64_t target = ++m_computeFenceValue;
    HRESULT hr = m_computeQueue->Signal(m_computeFence.Get(), target);
    if (FAILED(hr)) {
        m_gpuFaulted = true;
        return hr;
    }

    uint32_t waitedMs = 0;
    while (m_computeFence->GetCompletedValue() < target) {
        hr = m_computeFence->SetEventOnCompletion(target, m_computeFenceEvent);
        if (FAILED(hr)) return hr;
        const DWORD waitResult = WaitForSingleObject(m_computeFenceEvent, 250);
        if (waitResult == WAIT_OBJECT_0) break;
        if (waitResult == WAIT_FAILED) return E_FAIL;
        const HRESULT removed = m_d3dDevice->GetDeviceRemovedReason();
        if (FAILED(removed)) {
            m_gpuFaulted = true;
            return removed;
        }
        waitedMs += 250;
        if (waitedMs >= kMaxGpuWaitMs) {
            m_gpuFaulted = true;
            return HRESULT_FROM_WIN32(ERROR_TIMEOUT);
        }
    }

    m_inFlight = false;
    const HRESULT removed = m_d3dDevice->GetDeviceRemovedReason();
    if (FAILED(removed)) {
        m_gpuFaulted = true;
        return removed;
    }
    return S_OK;
}

HRESULT FMBrain::GpuBegin() {
    if (!m_computeAlloc || !m_computeCmdList || !m_rootSig) return E_NOT_VALID_STATE;
    if (m_inFlight) {
        // Never reset an allocator the GPU may still be reading.
        const HRESULT waitHr = WaitComputeQueue();
        if (FAILED(waitHr)) return waitHr;
    }
    if (m_recording) GpuAbort(); // a previous recording was never submitted; start clean

    HRESULT hr = m_computeAlloc->Reset();
    if (FAILED(hr)) return hr;
    hr = m_computeCmdList->Reset(m_computeAlloc.Get(), nullptr);
    if (FAILED(hr)) return hr;
    m_computeCmdList->SetComputeRootSignature(m_rootSig.Get());
    m_recording = true;
    m_pendingWork = 0.0;
    return S_OK;
}

HRESULT FMBrain::GpuFlush(bool reopen) {
    if (!m_recording) return E_NOT_VALID_STATE;
    const HRESULT closeHr = m_computeCmdList->Close();
    m_recording = false;
    if (FAILED(closeHr)) {
        m_gpuLayoutValid = false; // recorded barriers will never execute: buffer states are no longer trustworthy
        return closeHr;
    }
    ID3D12CommandList* lists[] = { m_computeCmdList.Get() };
    m_computeQueue->ExecuteCommandLists(1, lists);
    m_inFlight = true;
    if (!reopen) return S_OK;

    // A single allocator cannot be reset while the GPU still runs its list: wait, then reopen.
    HRESULT hr = WaitComputeQueue();
    if (FAILED(hr)) return hr;
    return GpuBegin();
}

HRESULT FMBrain::GpuFinishAndWait() {
    HRESULT hr = GpuFlush(false);
    if (FAILED(hr)) return hr;
    return WaitComputeQueue();
}

void FMBrain::GpuAbort() {
    if (m_recording && m_computeCmdList) {
        m_computeCmdList->Close();
    }
    m_recording = false;
    // Barriers recorded in the abandoned list were never executed, so tracked resource states may be wrong.
    // Force a clean rebuild (fresh buffers in known states + re-upload from the CPU copy) before the next use.
    m_gpuLayoutValid = false;
}

void FMBrain::TransitionTo(GpuBuffer& buf, D3D12_RESOURCE_STATES state) {
    if (buf.fixedState || buf.state == state) return;
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    b.Transition.pResource = buf.res.Get();
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = buf.state;
    b.Transition.StateAfter = state;
    m_computeCmdList->ResourceBarrier(1, &b);
    buf.state = state;
    buf.uavDirty = false;
}

void FMBrain::RecordUploadCopy(GpuBuffer& dst, uint64_t dstOffset, uint64_t uploadOffset, uint64_t bytes) {
    TransitionTo(dst, D3D12_RESOURCE_STATE_COPY_DEST);
    m_computeCmdList->CopyBufferRegion(dst.res.Get(), dstOffset, m_act.upload.res.Get(), uploadOffset, bytes);
}

void FMBrain::RecordReadbackCopy(GpuBuffer& src, uint64_t srcOffset, uint64_t readbackOffset, uint64_t bytes) {
    TransitionTo(src, D3D12_RESOURCE_STATE_COPY_SOURCE);
    m_computeCmdList->CopyBufferRegion(m_act.readback.res.Get(), readbackOffset, src.res.Get(), srcOffset, bytes);
}

HRESULT FMBrain::RecordDispatch(const GpuDispatch& d) {
    if (!m_recording) return E_NOT_VALID_STATE;
    if (d.gx == 0 || d.gy == 0 || d.gz == 0) return S_OK;
    if (d.gx > kMaxDispatchDim || d.gy > kMaxDispatchDim || d.gz > kMaxDispatchDim) {
        GpuLog("refused an oversized dispatch (" + std::to_string(d.gx) + " x " + std::to_string(d.gy) + " x " +
               std::to_string(d.gz) + " groups) for kernel " +
               kKernelFiles[static_cast<uint32_t>(d.kernel)].label);
        return E_INVALIDARG;
    }
    const uint32_t kernelIndex = static_cast<uint32_t>(d.kernel);
    if (kernelIndex >= kGpuKernelCount || !m_pso[kernelIndex]) return E_NOT_VALID_STATE;

    // 1) Which state must each distinct buffer be in for this dispatch?
    struct Need {
        GpuBuffer* buf;
        D3D12_RESOURCE_STATES state;
    };
    Need needs[7];
    uint32_t needCount = 0;
    auto addNeed = [&](GpuBuffer* b, D3D12_RESOURCE_STATES s) -> bool {
        if (!b) return true;
        for (uint32_t i = 0; i < needCount; ++i) {
            if (needs[i].buf == b) return needs[i].state == s; // same buffer, two states in one dispatch = bug
        }
        needs[needCount].buf = b;
        needs[needCount].state = s;
        ++needCount;
        return true;
    };
    for (const GpuBinding& s : d.srv) {
        if (!addNeed(s.buf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)) return E_INVALIDARG;
    }
    for (const GpuBinding& u : d.uav) {
        if (!addNeed(u.buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return E_INVALIDARG;
    }

    // 2) Transition barriers where the state differs, a UAV barrier where the previous dispatch wrote the buffer.
    D3D12_RESOURCE_BARRIER barriers[8];
    uint32_t barrierCount = 0;
    for (uint32_t i = 0; i < needCount; ++i) {
        GpuBuffer* b = needs[i].buf;
        D3D12_RESOURCE_BARRIER barrier = {};
        if (b->state != needs[i].state) {
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = b->res.Get();
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = b->state;
            barrier.Transition.StateAfter = needs[i].state;
            barriers[barrierCount++] = barrier;
            b->state = needs[i].state;
            b->uavDirty = false;
        } else if (needs[i].state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS && b->uavDirty) {
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.UAV.pResource = b->res.Get();
            barriers[barrierCount++] = barrier;
            b->uavDirty = false;
        }
    }
    if (barrierCount > 0) {
        m_computeCmdList->ResourceBarrier(barrierCount, barriers);
    }

    // 3) Bind and launch. Every root slot is given a valid address: a slot this kernel does not use is pointed at a
    //    buffer that is already bound (so it is already in the right state) and the shader never touches it.
    //    That way no root argument is ever left unset, which strict D3D12 validation can complain about.
    m_computeCmdList->SetPipelineState(m_pso[kernelIndex].Get());
    m_computeCmdList->SetComputeRoot32BitConstants(0, kConstantDwords, &d.cb, 0);
    const GpuBinding* firstSrv = nullptr;
    for (const GpuBinding& s : d.srv) {
        if (s.buf) { firstSrv = &s; break; }
    }
    const GpuBinding* firstUav = nullptr;
    for (const GpuBinding& u : d.uav) {
        if (u.buf) { firstUav = &u; break; }
    }
    for (UINT i = 0; i < 4; ++i) {
        const GpuBinding* s = d.srv[i].buf ? &d.srv[i] : firstSrv;
        if (s) {
            m_computeCmdList->SetComputeRootShaderResourceView(
                1 + i, s->buf->res->GetGPUVirtualAddress() + s->byteOffset);
        }
    }
    for (UINT i = 0; i < 3; ++i) {
        const GpuBinding* u = d.uav[i].buf ? &d.uav[i] : firstUav;
        if (u) {
            m_computeCmdList->SetComputeRootUnorderedAccessView(
                5 + i, u->buf->res->GetGPUVirtualAddress() + u->byteOffset);
        }
    }
    m_computeCmdList->Dispatch(d.gx, d.gy, d.gz);
    for (const GpuBinding& u : d.uav) {
        if (u.buf) u.buf->uavDirty = true;
    }

    // 4) Keep every submission short: long steps are cut into several submits so no single one can hit a GPU timeout.
    m_pendingWork += d.work;
    if (m_pendingWork >= kSubmitWorkThreshold) {
        return GpuFlush(true);
    }
    return S_OK;
}

HRESULT FMBrain::NoteGpuFailure(const char* stage, HRESULT hr) {
    if (m_d3dDevice) {
        const HRESULT removed = m_d3dDevice->GetDeviceRemovedReason();
        if (FAILED(removed)) {
            m_gpuFaulted = true;
            if (m_failureLogCount < kMaxFailureLogLines) {
                GpuLog("device removed reason " + HexHr(removed));
            }
        }
    }
    if (m_failureLogCount < kMaxFailureLogLines) {
        ++m_failureLogCount;
        GpuLog(std::string("GPU step failed while ") + stage + " (HRESULT " + HexHr(hr) + ")");
    }
    if (m_gpuFaulted && m_gpuReady) {
        m_gpuReady = false;
        GpuLog("GPU marked unusable until it is re-initialised; training continues on the CPU.");
    }
    return hr;
}

// =================================================================================================
// GPU: recorded passes
// =================================================================================================

HRESULT FMBrain::RecordForwardPass(uint32_t batch, uint32_t height, uint32_t width, bool useConstT, float constT) {
    const uint32_t hw = height * width;
    const uint32_t blocks = m_arch.numResBlocks;
    const uint32_t cIn = m_arch.TotalInputChannels();
    const uint32_t imgC = m_arch.inputImageChannels;
    const GPUTrainConstants base = BaseConstants(batch, height, width, imgC);
    HRESULT hr = S_OK;

    // 1) sine-operator encoding (gInC = image channels of x_t, gOutC = channels of the encoded input)
    {
        GpuDispatch d;
        d.kernel = GpuKernel::EncodeSine;
        d.cb = base;
        d.cb.gInC = imgC;
        d.cb.gOutC = cIn;
        d.cb.gUseConstT = useConstT ? 1u : 0u;
        d.cb.gTimeT = constT;
        d.srv[0] = { &m_act.xt, 0 };
        if (!useConstT) d.srv[1] = { &m_act.tValues, 0 };
        d.uav[0] = { &m_act.stemIn, 0 };
        SetImageGrid(d, width, height, batch);
        d.work = static_cast<double>(batch) * hw * cIn;
        if (FAILED(hr = RecordDispatch(d))) return hr;
    }

    // One fused conv launch: out = [SiLU](conv(in)) [+ residual]
    auto layerFwd = [&](size_t li, GpuBuffer* in, GpuBuffer* out, GpuBuffer* pre, GpuBuffer* residual, bool silu) -> HRESULT {
        const GpuLayerInfo& L = m_gpuLayers[li];
        GpuDispatch d;
        d.kernel = GpuKernel::ConvFwd;
        d.cb = base;
        d.cb.gInC = L.inC;
        d.cb.gOutC = L.outC;
        d.cb.gKernelSize = L.k;
        d.cb.gApplySiLU = silu ? 1u : 0u;
        d.cb.gAddResidual = residual ? 1u : 0u;
        d.srv[0] = { in, 0 };
        d.srv[1] = { &m_gpuParams, L.wOff * sizeof(float) };
        d.srv[2] = { &m_gpuParams, L.bOff * sizeof(float) };
        if (residual) d.srv[3] = { residual, 0 };
        d.uav[0] = { out, 0 };
        if (silu) d.uav[1] = { pre, 0 };
        SetImageGrid(d, width, height, batch * L.outC);
        d.work = static_cast<double>(batch) * L.outC * hw * L.inC * L.k * L.k;
        return RecordDispatch(d);
    };

    // 2) stem (1x1 conv + SiLU), 3) residual blocks, 4) velocity head (1x1 conv, linear)
    if (FAILED(hr = layerFwd(0, &m_act.stemIn, &m_act.act[0], &m_act.stemPre, nullptr, true))) return hr;
    for (uint32_t b = 0; b < blocks; ++b) {
        const size_t c1 = 1 + 2 * static_cast<size_t>(b);
        const size_t c2 = c1 + 1;
        if (FAILED(hr = layerFwd(c1, &m_act.act[b], &m_act.midAct[b], &m_act.midPre[b], nullptr, true))) return hr;
        // conv2 is linear and the skip connection is added inside the same launch
        if (FAILED(hr = layerFwd(c2, &m_act.midAct[b], &m_act.act[b + 1], nullptr, &m_act.act[b], false))) return hr;
    }
    const size_t headIdx = 1 + 2 * static_cast<size_t>(blocks);
    return layerFwd(headIdx, &m_act.act[blocks], &m_act.vPred, nullptr, nullptr, false);
}

HRESULT FMBrain::RecordBackwardPass(uint32_t batch, uint32_t height, uint32_t width) {
    const uint32_t hw = height * width;
    const uint32_t blocks = m_arch.numResBlocks;
    const uint64_t rgbCount = static_cast<uint64_t>(batch) * m_arch.inputImageChannels * hw;
    const GPUTrainConstants base = BaseConstants(batch, height, width, m_arch.inputImageChannels);
    HRESULT hr = S_OK;

    // MSE gradient: dL/dv = 2 * (v_pred - v_target) / (N*C*H*W)   (C = 3 for pixels, 12 for the wave form)
    {
        GpuDispatch d;
        d.kernel = GpuKernel::MseGrad;
        d.cb = base;
        d.cb.gParamCount = static_cast<uint32_t>(rgbCount);
        d.cb.gInvOutCount = 1.0f / static_cast<float>(std::max<uint64_t>(1, rgbCount));
        d.srv[0] = { &m_act.vPred, 0 };
        d.srv[1] = { &m_act.vTarget, 0 };
        d.uav[0] = { &m_act.gradPred, 0 };
        SetLinearGrid(d, rgbCount);
        d.work = static_cast<double>(rgbCount);
        if (FAILED(hr = RecordDispatch(d))) return hr;
    }

    // weight / bias gradient of one layer: one thread group per (outC, inC) pair
    auto bwdWeights = [&](size_t li, GpuBuffer* in, GpuBuffer* gradOut) -> HRESULT {
        const GpuLayerInfo& L = m_gpuLayers[li];
        GpuDispatch d;
        d.kernel = GpuKernel::ConvBwdW;
        d.cb = base;
        d.cb.gInC = L.inC;
        d.cb.gOutC = L.outC;
        d.cb.gKernelSize = L.k;
        d.srv[0] = { in, 0 };
        d.srv[1] = { gradOut, 0 };
        d.uav[1] = { &m_gpuGrads, L.wOff * sizeof(float) };
        d.uav[2] = { &m_gpuGrads, L.bOff * sizeof(float) };
        d.gx = L.inC * L.outC;
        d.gy = 1;
        d.gz = 1;
        d.work = static_cast<double>(L.wCount) * batch * hw;
        return RecordDispatch(d);
    };

    // gradient w.r.t. a layer's input, optionally fused with "+ residual" and/or "* SiLU'(preAct)"
    auto bwdInput = [&](size_t li, GpuBuffer* gradOut, GpuBuffer* gradIn, GpuBuffer* preAct, GpuBuffer* residual) -> HRESULT {
        const GpuLayerInfo& L = m_gpuLayers[li];
        GpuDispatch d;
        d.kernel = GpuKernel::ConvBwdIn;
        d.cb = base;
        d.cb.gInC = L.inC;
        d.cb.gOutC = L.outC;
        d.cb.gKernelSize = L.k;
        d.cb.gSiLUGradEpi = preAct ? 1u : 0u;
        d.cb.gAddResidual = residual ? 1u : 0u;
        d.srv[0] = { gradOut, 0 };
        d.srv[1] = { &m_gpuParams, L.wOff * sizeof(float) };
        if (preAct) d.srv[2] = { preAct, 0 };
        if (residual) d.srv[3] = { residual, 0 };
        d.uav[0] = { gradIn, 0 };
        SetImageGrid(d, width, height, batch * L.inC);
        d.work = static_cast<double>(batch) * L.inC * hw * L.outC * L.k * L.k;
        return RecordDispatch(d);
    };

    // Head
    const size_t headIdx = 1 + 2 * static_cast<size_t>(blocks);
    if (FAILED(hr = bwdWeights(headIdx, &m_act.act[blocks], &m_act.gradPred))) return hr;
    if (FAILED(hr = bwdInput(headIdx, &m_act.gradPred, &m_act.gradA, nullptr, nullptr))) return hr;

    // Residual blocks, last to first.  Forward: out = Conv2(SiLU(Conv1(in))) + in
    //   gradW2/gradB2 from (midAct, gradCur);  gradMidPre = Conv2_BwdIn(gradCur) * SiLU'(midPre)   [fused]
    //   gradW1/gradB1 from (blockIn, gradMidPre);  gradIn = Conv1_BwdIn(gradMidPre) + gradCur      [fused skip]
    GpuBuffer* gradCur = &m_act.gradA;
    GpuBuffer* gradNext = &m_act.gradB;
    for (uint32_t b = blocks; b-- > 0;) {
        const size_t c1 = 1 + 2 * static_cast<size_t>(b);
        const size_t c2 = c1 + 1;
        if (FAILED(hr = bwdWeights(c2, &m_act.midAct[b], gradCur))) return hr;
        if (FAILED(hr = bwdInput(c2, gradCur, &m_act.gradTmp, &m_act.midPre[b], nullptr))) return hr;
        if (FAILED(hr = bwdWeights(c1, &m_act.act[b], &m_act.gradTmp))) return hr;
        if (FAILED(hr = bwdInput(c1, &m_act.gradTmp, gradNext, nullptr, gradCur))) return hr;
        std::swap(gradCur, gradNext);
    }

    // Stem (had a SiLU): gradPre = gradCur * SiLU'(stemPre), then the weight gradient. No input gradient is needed.
    {
        GpuDispatch d;
        d.kernel = GpuKernel::SiLUBwd;
        d.cb = base;
        d.cb.gParamCount = static_cast<uint32_t>(static_cast<uint64_t>(batch) * m_arch.width * hw);
        d.srv[0] = { gradCur, 0 };
        d.srv[1] = { &m_act.stemPre, 0 };
        d.uav[0] = { &m_act.gradTmp, 0 };
        SetLinearGrid(d, d.cb.gParamCount);
        d.work = static_cast<double>(d.cb.gParamCount);
        if (FAILED(hr = RecordDispatch(d))) return hr;
    }
    return bwdWeights(0, &m_act.stemIn, &m_act.gradTmp);
}

HRESULT FMBrain::RecordOptimizerPass(uint32_t batch, uint32_t height, uint32_t width, uint64_t stepIndex,
                                     float lr, float beta1, float beta2, float eps, float gradClipNorm) {
    const uint64_t rgbCount = static_cast<uint64_t>(batch) * m_arch.inputImageChannels * height * width;
    const GPUTrainConstants base = BaseConstants(batch, height, width, m_arch.inputImageChannels);
    HRESULT hr = S_OK;

    // Partial sums of squares: the gradient arena (-> norm for clipping) and gradPred (-> the MSE loss).
    auto sumSquares = [&](GpuBuffer* src, uint64_t count, uint32_t partialOffset) -> HRESULT {
        GpuDispatch d;
        d.kernel = GpuKernel::SumSquares;
        d.cb = base;
        d.cb.gParamCount = static_cast<uint32_t>(count);
        d.cb.gPartialCount = kReduceThreads;
        d.cb.gPartialOffset = partialOffset;
        d.srv[0] = { src, 0 };
        d.uav[0] = { &m_gpuPartials, 0 };
        d.gx = kReduceThreads / 64;
        d.gy = 1;
        d.gz = 1;
        d.work = static_cast<double>(count);
        return RecordDispatch(d);
    };
    if (FAILED(hr = sumSquares(&m_gpuGrads, m_arenaFloats, 0))) return hr;
    if (FAILED(hr = sumSquares(&m_act.gradPred, rgbCount, kReduceThreads))) return hr;

    // Final reduction + clip scale + loss
    {
        GpuDispatch d;
        d.kernel = GpuKernel::ReduceFinal;
        d.cb = base;
        d.cb.gPartialCount = kReduceThreads;
        d.cb.gInvOutCount = 1.0f / static_cast<float>(std::max<uint64_t>(1, rgbCount));
        d.cb.gGradClipNorm = gradClipNorm;
        d.srv[0] = { &m_gpuPartials, 0 };
        d.uav[0] = { &m_gpuStats, 0 };
        d.gx = 1;
        d.gy = 1;
        d.gz = 1;
        d.work = static_cast<double>(2 * kReduceThreads);
        if (FAILED(hr = RecordDispatch(d))) return hr;
    }

    // Adam over the whole parameter arena (skipped on the GPU if the gradient was NaN / Inf)
    {
        GpuDispatch d;
        d.kernel = GpuKernel::Adam;
        d.cb = base;
        d.cb.gParamCount = static_cast<uint32_t>(m_arenaFloats);
        d.cb.gLearningRate = lr;
        d.cb.gBeta1 = beta1;
        d.cb.gBeta2 = beta2;
        d.cb.gEpsilon = eps;
        d.cb.gBiasCorr1 = 1.0f - std::pow(beta1, static_cast<float>(stepIndex));
        d.cb.gBiasCorr2 = 1.0f - std::pow(beta2, static_cast<float>(stepIndex));
        d.srv[0] = { &m_gpuGrads, 0 };
        d.srv[1] = { &m_gpuStats, 0 };
        d.uav[0] = { &m_gpuParams, 0 };
        d.uav[1] = { &m_gpuAdamM, 0 };
        d.uav[2] = { &m_gpuAdamV, 0 };
        SetLinearGrid(d, m_arenaFloats);
        d.work = static_cast<double>(m_arenaFloats) * 4.0;
        if (FAILED(hr = RecordDispatch(d))) return hr;
    }
    return S_OK;
}

// =================================================================================================
// GPU: weights up / down
// =================================================================================================

HRESULT FMBrain::UploadWeightsToGPU() {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    return UploadWeightsToGPULocked();
}

HRESULT FMBrain::UploadWeightsToGPULocked() {
    if (!m_d3dDevice || !m_computeQueue || !m_rootSig) return E_NOT_VALID_STATE;

    HRESULT hr = S_OK;
    if (!GpuLayoutMatchesLocked()) {
        hr = RebuildGpuLayoutLocked();
        if (FAILED(hr)) return hr;
    }
    // The staging buffers live with the activation set, which is freed if a resize ever failed: make sure they exist.
    hr = EnsureActivationBuffersLocked(kMinActivationPixels);
    if (FAILED(hr)) return hr;

    const std::vector<LayerVecs> layers = CollectLayers(*this);
    if (layers.size() != m_gpuLayers.size()) return E_NOT_VALID_STATE;
    for (size_t i = 0; i < layers.size(); ++i) {
        const GpuLayerInfo& L = m_gpuLayers[i];
        const LayerVecs& V = layers[i];
        if (V.layer->weight.size() != L.wCount || V.layer->bias.size() != L.bCount ||
            V.mW->size() != L.wCount || V.vW->size() != L.wCount ||
            V.mB->size() != L.bCount || V.vB->size() != L.bCount) {
            GpuLog("weight upload refused: CPU layer " + std::to_string(i) + " does not match the GPU layout");
            return E_NOT_VALID_STATE;
        }
    }

    hr = GpuBegin();
    if (FAILED(hr)) return hr;

    uint8_t* mapped = nullptr;
    hr = m_act.upload.res->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    if (FAILED(hr)) {
        GpuAbort();
        return hr;
    }

    // Three images back to back: parameters | Adam m | Adam v, each exactly one arena long.
    std::memset(mapped, 0, static_cast<size_t>(3 * m_arenaBytes));
    float* paramImg = reinterpret_cast<float*>(mapped);
    float* mImg = paramImg + m_arenaFloats;
    float* vImg = mImg + m_arenaFloats;
    for (size_t i = 0; i < layers.size(); ++i) {
        const GpuLayerInfo& L = m_gpuLayers[i];
        const LayerVecs& V = layers[i];
        std::memcpy(paramImg + L.wOff, V.layer->weight.data(), static_cast<size_t>(L.wCount) * sizeof(float));
        std::memcpy(paramImg + L.bOff, V.layer->bias.data(), static_cast<size_t>(L.bCount) * sizeof(float));
        std::memcpy(mImg + L.wOff, V.mW->data(), static_cast<size_t>(L.wCount) * sizeof(float));
        std::memcpy(mImg + L.bOff, V.mB->data(), static_cast<size_t>(L.bCount) * sizeof(float));
        std::memcpy(vImg + L.wOff, V.vW->data(), static_cast<size_t>(L.wCount) * sizeof(float));
        std::memcpy(vImg + L.bOff, V.vB->data(), static_cast<size_t>(L.bCount) * sizeof(float));
    }
    m_act.upload.res->Unmap(0, nullptr);

    RecordUploadCopy(m_gpuParams, 0, 0, m_arenaBytes);
    RecordUploadCopy(m_gpuAdamM, 0, m_arenaBytes, m_arenaBytes);
    RecordUploadCopy(m_gpuAdamV, 0, 2 * m_arenaBytes, m_arenaBytes);

    hr = GpuFinishAndWait();
    if (FAILED(hr)) return hr;

    m_gpuNeedsUpload = false;
    m_gpuWeightsDirtyOnDevice = false;
    m_lastWeightSync = std::chrono::steady_clock::now();
    return S_OK;
}

HRESULT FMBrain::ReadbackWeightsFromGPU() {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    return ReadbackWeightsFromGPULocked();
}

HRESULT FMBrain::ReadbackWeightsFromGPULocked() {
    // Nothing to do unless the GPU holds newer weights than the CPU vectors.
    if (!m_gpuReady || !m_gpuWeightsDirtyOnDevice || m_gpuNeedsUpload) return S_OK;
    if (!GpuLayoutMatchesLocked()) return S_OK; // the CPU copy is the authority for a different / rebuilt model

    HRESULT hr = EnsureActivationBuffersLocked(kMinActivationPixels); // the readback staging buffer lives there
    if (FAILED(hr)) return hr;
    hr = GpuBegin();
    if (FAILED(hr)) return hr;
    RecordReadbackCopy(m_gpuParams, 0, 0, m_arenaBytes);
    RecordReadbackCopy(m_gpuAdamM, 0, m_arenaBytes, m_arenaBytes);
    RecordReadbackCopy(m_gpuAdamV, 0, 2 * m_arenaBytes, m_arenaBytes);
    hr = GpuFinishAndWait();
    if (FAILED(hr)) return hr;

    uint8_t* mapped = nullptr;
    D3D12_RANGE range;
    range.Begin = 0;
    range.End = static_cast<SIZE_T>(3 * m_arenaBytes);
    hr = m_act.readback.res->Map(0, &range, reinterpret_cast<void**>(&mapped));
    if (FAILED(hr)) return hr;

    const std::vector<LayerVecs> layers = CollectLayers(*this);
    const float* paramImg = reinterpret_cast<const float*>(mapped);
    const float* mImg = paramImg + m_arenaFloats;
    const float* vImg = mImg + m_arenaFloats;
    if (layers.size() == m_gpuLayers.size()) {
        for (size_t i = 0; i < layers.size(); ++i) {
            const GpuLayerInfo& L = m_gpuLayers[i];
            const LayerVecs& V = layers[i];
            std::memcpy(V.layer->weight.data(), paramImg + L.wOff, static_cast<size_t>(L.wCount) * sizeof(float));
            std::memcpy(V.layer->bias.data(), paramImg + L.bOff, static_cast<size_t>(L.bCount) * sizeof(float));
            std::memcpy(V.mW->data(), mImg + L.wOff, static_cast<size_t>(L.wCount) * sizeof(float));
            std::memcpy(V.mB->data(), mImg + L.bOff, static_cast<size_t>(L.bCount) * sizeof(float));
            std::memcpy(V.vW->data(), vImg + L.wOff, static_cast<size_t>(L.wCount) * sizeof(float));
            std::memcpy(V.vB->data(), vImg + L.bOff, static_cast<size_t>(L.bCount) * sizeof(float));
        }
    }
    D3D12_RANGE emptyRange;
    emptyRange.Begin = 0;
    emptyRange.End = 0;
    m_act.readback.res->Unmap(0, &emptyRange);

    m_gpuWeightsDirtyOnDevice = false;
    m_lastWeightSync = std::chrono::steady_clock::now();
    return S_OK;
}

// =================================================================================================
// GPU: training step
// =================================================================================================

HRESULT FMBrain::TrainStepGpuLocked(
    const float* xt, const float* vTarget, const float* tValues,
    uint32_t batch, uint32_t height, uint32_t width, uint64_t stepIndex,
    float lr, float beta1, float beta2, float eps, float gradClipNorm, float* outStats) {
    const uint32_t hw = height * width;
    const uint64_t rgbBytes = static_cast<uint64_t>(batch) * m_arch.inputImageChannels * hw * sizeof(float);
    const uint64_t tBytes = static_cast<uint64_t>(batch) * sizeof(float);
    const uint64_t offXt = 0;
    const uint64_t offVt = AlignUp(rgbBytes, 256);
    const uint64_t offT = AlignUp(offVt + rgbBytes, 256);
    if (offT + tBytes > m_act.upload.bytes) return E_OUTOFMEMORY; // cannot happen: sized from capacityPixels

    HRESULT hr = GpuBegin();
    if (FAILED(hr)) return hr;

    // Only this lesson's small batch goes up; the weights stay in GPU memory.
    uint8_t* mapped = nullptr;
    hr = m_act.upload.res->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    if (FAILED(hr)) {
        GpuAbort();
        return hr;
    }
    std::memcpy(mapped + offXt, xt, static_cast<size_t>(rgbBytes));
    std::memcpy(mapped + offVt, vTarget, static_cast<size_t>(rgbBytes));
    std::memcpy(mapped + offT, tValues, static_cast<size_t>(tBytes));
    m_act.upload.res->Unmap(0, nullptr);

    RecordUploadCopy(m_act.xt, 0, offXt, rgbBytes);
    RecordUploadCopy(m_act.vTarget, 0, offVt, rgbBytes);
    RecordUploadCopy(m_act.tValues, 0, offT, tBytes);

    hr = RecordForwardPass(batch, height, width, false, 0.0f);
    if (SUCCEEDED(hr)) hr = RecordBackwardPass(batch, height, width);
    if (SUCCEEDED(hr)) hr = RecordOptimizerPass(batch, height, width, stepIndex, lr, beta1, beta2, eps, gradClipNorm);
    if (FAILED(hr)) {
        GpuAbort();
        return hr;
    }
    if (outStats) {
        RecordReadbackCopy(m_gpuStats, 0, 0, 4 * sizeof(float));
    }

    hr = GpuFinishAndWait();
    if (FAILED(hr)) return hr;

    if (outStats) {
        uint8_t* rb = nullptr;
        D3D12_RANGE range;
        range.Begin = 0;
        range.End = 4 * sizeof(float);
        hr = m_act.readback.res->Map(0, &range, reinterpret_cast<void**>(&rb));
        if (FAILED(hr)) return hr;
        std::memcpy(outStats, rb, 4 * sizeof(float));
        D3D12_RANGE emptyRange;
        emptyRange.Begin = 0;
        emptyRange.End = 0;
        m_act.readback.res->Unmap(0, &emptyRange);
    }
    return S_OK;
}

HRESULT FMBrain::TrainStepGPU(
    const float* xt,
    const float* vTarget,
    const float* tValues,
    uint32_t batch,
    uint32_t height,
    uint32_t width,
    uint64_t stepIndex,
    float lr,
    float beta1,
    float beta2,
    float eps,
    bool computeLossReadback,
    float& outMseLoss,
    float gradClipNorm) {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    if (!m_gpuReady || m_gpuFaulted) return E_NOT_VALID_STATE;
    if (!xt || !vTarget || !tValues || batch == 0 || height == 0 || width == 0 || height > 128u || width > 128u) {
        return E_INVALIDARG;
    }

    // Never larger than the activation budget / tValues capacity; tiny at 128x128 so one submit can never run long.
    uint32_t useBatch = std::min(batch, 64u);
    useBatch = std::min(useBatch, MaxGpuBatchForResolution(std::max(height, width)));
    if (useBatch < batch && m_loggedBatchClampRes != std::max(height, width)) {
        m_loggedBatchClampRes = std::max(height, width);
        GpuLog("batch " + std::to_string(batch) + " reduced to " + std::to_string(useBatch) + " at " +
               std::to_string(width) + "x" + std::to_string(height) + " to stay inside the GPU memory budget");
    }

    const auto t0 = std::chrono::steady_clock::now();

    HRESULT hr = EnsureGpuInSyncLocked();
    if (FAILED(hr)) return NoteGpuFailure("syncing the weights to the GPU", hr);

    const uint32_t hw = height * width;
    for (;;) {
        hr = EnsureActivationBuffersLocked(static_cast<uint64_t>(useBatch) * hw);
        if (SUCCEEDED(hr) || hr != E_OUTOFMEMORY || useBatch <= 1) break;
        useBatch = (useBatch + 1) / 2; // out of memory: halve the batch and try again
    }
    if (FAILED(hr)) return NoteGpuFailure("allocating activation buffers", hr);

    float stats[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    hr = TrainStepGpuLocked(xt, vTarget, tValues, useBatch, height, width, stepIndex,
                            lr, beta1, beta2, eps, gradClipNorm, computeLossReadback ? stats : nullptr);
    if (FAILED(hr)) return NoteGpuFailure("running the training step", hr);

    m_gpuWeightsDirtyOnDevice = true;
    if (computeLossReadback) {
        outMseLoss = stats[2];
    }

    const auto t1 = std::chrono::steady_clock::now();
    if (!m_loggedFirstStep) {
        m_loggedFirstStep = true;
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        char buf[160] = {};
        std::snprintf(buf, sizeof(buf), "first GPU training step OK: batch %u at %ux%u took %.1f ms (loss %.5f)",
                      useBatch, width, height, ms, static_cast<double>(stats[2]));
        GpuLog(buf);
    }

    // Safety checkpoint: copy the GPU weights back every 30 s so a GPU crash never loses more than that.
    if (t1 - m_lastWeightSync > std::chrono::seconds(30)) {
        const HRESULT rh = ReadbackWeightsFromGPULocked();
        if (FAILED(rh)) {
            GpuLog("periodic weight checkpoint failed (HRESULT " + HexHr(rh) + ")");
        }
        m_lastWeightSync = std::chrono::steady_clock::now();
    }
    return S_OK;
}

// =================================================================================================
// GPU: forward / dream
// =================================================================================================

HRESULT FMBrain::RunForwardGpuLocked(const float* xt, const float* tValues, uint32_t batch, uint32_t height,
                                     uint32_t width, std::vector<float>& outVelocity) {
    const uint32_t hw = height * width;
    HRESULT hr = EnsureActivationBuffersLocked(static_cast<uint64_t>(batch) * hw);
    if (FAILED(hr)) return hr;

    const uint64_t rgbBytes = static_cast<uint64_t>(batch) * m_arch.inputImageChannels * hw * sizeof(float);
    const uint64_t tBytes = static_cast<uint64_t>(batch) * sizeof(float);
    const uint64_t offT = AlignUp(rgbBytes, 256);

    hr = GpuBegin();
    if (FAILED(hr)) return hr;

    uint8_t* mapped = nullptr;
    hr = m_act.upload.res->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    if (FAILED(hr)) {
        GpuAbort();
        return hr;
    }
    std::memcpy(mapped, xt, static_cast<size_t>(rgbBytes));
    std::memcpy(mapped + offT, tValues, static_cast<size_t>(tBytes));
    m_act.upload.res->Unmap(0, nullptr);
    RecordUploadCopy(m_act.xt, 0, 0, rgbBytes);
    RecordUploadCopy(m_act.tValues, 0, offT, tBytes);

    hr = RecordForwardPass(batch, height, width, false, 0.0f);
    if (FAILED(hr)) {
        GpuAbort();
        return hr;
    }
    RecordReadbackCopy(m_act.vPred, 0, 0, rgbBytes);
    hr = GpuFinishAndWait();
    if (FAILED(hr)) return hr;

    uint8_t* rb = nullptr;
    D3D12_RANGE range;
    range.Begin = 0;
    range.End = static_cast<SIZE_T>(rgbBytes);
    hr = m_act.readback.res->Map(0, &range, reinterpret_cast<void**>(&rb));
    if (FAILED(hr)) return hr;
    outVelocity.resize(static_cast<size_t>(rgbBytes / sizeof(float)));
    std::memcpy(outVelocity.data(), rb, static_cast<size_t>(rgbBytes));
    D3D12_RANGE emptyRange;
    emptyRange.Begin = 0;
    emptyRange.End = 0;
    m_act.readback.res->Unmap(0, &emptyRange);
    return S_OK;
}

HRESULT FMBrain::DreamEulerODEGPU(
    const float* initialNoise,
    uint32_t height,
    uint32_t width,
    uint32_t steps,
    std::vector<float>& outImageCHW) {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    if (!m_gpuReady || m_gpuFaulted) return E_NOT_VALID_STATE;
    if (!initialNoise || height == 0 || width == 0 || height > 128u || width > 128u) return E_INVALIDARG;

    HRESULT hr = EnsureGpuInSyncLocked();
    if (FAILED(hr)) return NoteGpuFailure("syncing the weights for dreaming", hr);

    const uint32_t hw = height * width;
    hr = EnsureActivationBuffersLocked(hw);
    if (FAILED(hr)) return NoteGpuFailure("allocating dream buffers", hr);

    const uint32_t stepCount = std::max(1u, steps);
    const float dt = 1.0f / static_cast<float>(stepCount);
    const uint64_t sampleBytes = static_cast<uint64_t>(m_arch.inputImageChannels) * hw * sizeof(float);

    hr = GpuBegin();
    if (FAILED(hr)) return NoteGpuFailure("starting the dream", hr);

    uint8_t* mapped = nullptr;
    hr = m_act.upload.res->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    if (FAILED(hr)) {
        GpuAbort();
        return NoteGpuFailure("uploading the dream noise", hr);
    }
    std::memcpy(mapped, initialNoise, static_cast<size_t>(sampleBytes));
    m_act.upload.res->Unmap(0, nullptr);
    RecordUploadCopy(m_act.xt, 0, 0, sampleBytes);

    // The whole trajectory stays on the GPU: noise -> [forward -> Euler step] x N -> image
    for (uint32_t s = 0; s < stepCount && SUCCEEDED(hr); ++s) {
        // Euler: the network is asked about the picture at time s*dt (pure noise at s = 0). The old "+ 0.5" label made one- and
        // two-step dreams far worse (measured: 2.5x larger error at one step), so the label now matches the state it describes.
        const float t = static_cast<float>(s) * dt;
        hr = RecordForwardPass(1, height, width, true, t);
        if (FAILED(hr)) break;

        GpuDispatch d;
        d.kernel = GpuKernel::Euler;
        d.cb = BaseConstants(1, height, width, m_arch.inputImageChannels);
        d.cb.gParamCount = m_arch.inputImageChannels * hw;
        d.cb.gDt = dt;
        d.srv[0] = { &m_act.vPred, 0 };
        d.uav[0] = { &m_act.xt, 0 };
        SetLinearGrid(d, d.cb.gParamCount);
        d.work = static_cast<double>(d.cb.gParamCount);
        hr = RecordDispatch(d);
    }
    if (FAILED(hr)) {
        GpuAbort();
        return NoteGpuFailure("recording the dream", hr);
    }

    RecordReadbackCopy(m_act.xt, 0, 0, sampleBytes);
    hr = GpuFinishAndWait();
    if (FAILED(hr)) return NoteGpuFailure("running the dream", hr);

    uint8_t* rb = nullptr;
    D3D12_RANGE range;
    range.Begin = 0;
    range.End = static_cast<SIZE_T>(sampleBytes);
    hr = m_act.readback.res->Map(0, &range, reinterpret_cast<void**>(&rb));
    if (FAILED(hr)) return NoteGpuFailure("reading the dream back", hr);
    outImageCHW.resize(static_cast<size_t>(sampleBytes / sizeof(float)));
    std::memcpy(outImageCHW.data(), rb, static_cast<size_t>(sampleBytes));
    D3D12_RANGE emptyRange;
    emptyRange.Begin = 0;
    emptyRange.End = 0;
    m_act.readback.res->Unmap(0, &emptyRange);
    return S_OK;
}

// =================================================================================================
// GPU: self-test (runs inside InitializeGPU; the app only reports "GPU" if this passes)
// =================================================================================================

HRESULT FMBrain::SelfTestGpuLocked(std::string& report) {
    constexpr uint32_t N = 2;
    constexpr uint32_t H = 8;
    constexpr uint32_t W = 8;
    const size_t hw = static_cast<size_t>(H) * W;
    const size_t imgC = m_arch.inputImageChannels;

    std::vector<float> xt(N * imgC * hw);
    std::vector<float> target(N * imgC * hw);
    const float tv[N] = { 0.27f, 0.81f };
    for (size_t i = 0; i < xt.size(); ++i) {
        xt[i] = 0.8f * std::sin(0.37f * static_cast<float>(i) + 0.2f);
        target[i] = 0.7f * std::cos(0.23f * static_cast<float>(i) + 1.1f);
    }

    char buf[256] = {};

    // 1) Forward pass: GPU vs the trusted CPU implementation.
    ForwardCache cache;
    Forward(xt.data(), tv, N, H, W, cache);
    std::vector<float> gpuV;
    HRESULT hr = RunForwardGpuLocked(xt.data(), tv, N, H, W, gpuV);
    if (FAILED(hr)) {
        report = "self-test: GPU forward pass failed to run (HRESULT " + HexHr(hr) + ")";
        return hr;
    }
    if (gpuV.size() != cache.velocityPred.size()) {
        report = "self-test: GPU forward pass returned the wrong amount of data";
        return E_FAIL;
    }
    double maxDiff = 0.0;
    double cpuMse = 0.0;
    bool forwardOk = true;
    for (size_t i = 0; i < gpuV.size(); ++i) {
        const double ref = cache.velocityPred[i];
        const double got = gpuV[i];
        const double diff = std::abs(ref - got);
        if (!(diff <= 1e-3 + 1e-3 * std::abs(ref))) forwardOk = false; // also catches NaN
        maxDiff = std::max(maxDiff, diff);
        const double e = ref - static_cast<double>(target[i]);
        cpuMse += e * e;
    }
    cpuMse /= static_cast<double>(gpuV.size());
    if (!forwardOk) {
        std::snprintf(buf, sizeof(buf), "self-test FAILED: GPU forward pass disagrees with the CPU (max difference %.6g)", maxDiff);
        report = buf;
        return E_FAIL;
    }

    // 2) Two small training steps: the GPU loss must match the CPU loss, and must go down.
    float s1[4] = {};
    float s2[4] = {};
    hr = TrainStepGpuLocked(xt.data(), target.data(), tv, N, H, W, 1, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, s1);
    if (SUCCEEDED(hr)) {
        hr = TrainStepGpuLocked(xt.data(), target.data(), tv, N, H, W, 2, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, s2);
    }
    if (FAILED(hr)) {
        report = "self-test: GPU training step failed to run (HRESULT " + HexHr(hr) + ")";
        return hr;
    }
    const double lossErr = std::abs(static_cast<double>(s1[2]) - cpuMse);
    const bool lossOk = std::isfinite(s1[2]) && lossErr <= 1e-3 * std::max(1.0, cpuMse);
    const bool gradOk = std::isfinite(s1[3]) && s1[3] > 0.0f;
    const bool descentOk = std::isfinite(s2[2]) && s2[2] < s1[2];

    std::snprintf(buf, sizeof(buf),
                  "self-test: forward max diff %.3g | loss GPU %.6f vs CPU %.6f | grad norm %.4g | loss after 1 step %.6f -> %.6f",
                  maxDiff, static_cast<double>(s1[2]), cpuMse, static_cast<double>(s1[3]),
                  static_cast<double>(s1[2]), static_cast<double>(s2[2]));
    report = buf;

    // Put the CPU copy of weights + Adam moments back (the test steps changed the GPU copy).
    m_gpuNeedsUpload = true;
    const HRESULT restoreHr = UploadWeightsToGPULocked();
    if (FAILED(restoreHr)) {
        report += " | restoring the weights failed (HRESULT " + HexHr(restoreHr) + ")";
        return restoreHr;
    }

    if (!lossOk) {
        report += " | FAILED: GPU loss does not match the CPU loss";
        return E_FAIL;
    }
    if (!gradOk) {
        report += " | FAILED: gradient norm is zero or not a number";
        return E_FAIL;
    }
    if (!descentOk) {
        report += " | FAILED: loss did not go down after a training step";
        return E_FAIL;
    }
    report += " | PASSED";
    return S_OK;
}

// =================================================================================================
// CPU forward / dream
// =================================================================================================

void FMBrain::BuildSineOperatorInput(
    const float* xt,
    const float* tValues,
    uint32_t batch,
    uint32_t height,
    uint32_t width,
    std::vector<float>& outEncoded) const {
    const uint32_t cIn = m_arch.TotalInputChannels();
    const size_t imgC = m_arch.inputImageChannels; // 3 (pixels) or 12 (wave form): the image channels come first
    const size_t hw = static_cast<size_t>(height) * width;
    outEncoded.resize(static_cast<size_t>(batch) * cIn * hw);

    for (uint32_t n = 0; n < batch; ++n) {
        const float* xBatch = xt + static_cast<size_t>(n) * imgC * hw;
        float* dstBatch = outEncoded.data() + static_cast<size_t>(n) * cIn * hw;
        std::copy_n(xBatch, imgC * hw, dstBatch);

        const float t = tValues[n];
        const float tSin1 = std::sin(kPi * t);
        const float tCos1 = std::cos(kPi * t);
        const float tSin2 = std::sin(2.0f * kPi * t);
        const float tCos2 = std::cos(2.0f * kPi * t);

        for (uint32_t y = 0; y < height; ++y) {
            const float ny = (height > 1) ? (2.0f * static_cast<float>(y) / static_cast<float>(height - 1) - 1.0f) : 0.0f;
            for (uint32_t x = 0; x < width; ++x) {
                const float nx = (width > 1) ? (2.0f * static_cast<float>(x) / static_cast<float>(width - 1) - 1.0f) : 0.0f;
                const size_t p = static_cast<size_t>(y) * width + x;

                dstBatch[imgC * hw + p] = nx;
                dstBatch[(imgC + 1) * hw + p] = ny;

                size_t ch = imgC + 2;
                for (uint32_t k = 0; k < m_arch.harmonicBands; ++k) {
                    const float freq = static_cast<float>(1u << k) * kPi;
                    dstBatch[(ch + 0) * hw + p] = std::sin(freq * nx);
                    dstBatch[(ch + 1) * hw + p] = std::cos(freq * nx);
                    dstBatch[(ch + 2) * hw + p] = std::sin(freq * ny);
                    dstBatch[(ch + 3) * hw + p] = std::cos(freq * ny);
                    ch += 4;
                }

                dstBatch[(ch + 0) * hw + p] = tSin1;
                dstBatch[(ch + 1) * hw + p] = tCos1;
                dstBatch[(ch + 2) * hw + p] = tSin2;
                dstBatch[(ch + 3) * hw + p] = tCos2;
            }
        }
    }
}

void FMBrain::Forward(
    const float* xt,
    const float* tValues,
    uint32_t batch,
    uint32_t height,
    uint32_t width,
    ForwardCache& cache) const {
    cache.batch = batch;
    cache.height = height;
    cache.width = width;

    const uint32_t w = m_arch.width;
    const size_t hw = static_cast<size_t>(height) * width;
    const size_t hiddenSize = static_cast<size_t>(batch) * w * hw;

    BuildSineOperatorInput(xt, tValues, batch, height, width, cache.stemInput);

    cache.stemPreAct.resize(hiddenSize);
    cache.stemOut.resize(hiddenSize);
    Conv2dNCHW(cache.stemInput.data(), m_stem, batch, height, width, cache.stemPreAct.data());
    for (size_t i = 0; i < hiddenSize; ++i) {
        cache.stemOut[i] = SiLU(cache.stemPreAct[i]);
    }

    const size_t numBlocks = m_blocks.size();
    cache.blockIn.resize(numBlocks);
    cache.blockMidPre.resize(numBlocks);
    cache.blockMidAct.resize(numBlocks);
    cache.blockConv2Out.resize(numBlocks);

    const float* cur = cache.stemOut.data();
    std::vector<float> nextHidden(hiddenSize);

    for (size_t b = 0; b < numBlocks; ++b) {
        cache.blockIn[b].assign(cur, cur + hiddenSize);
        cache.blockMidPre[b].resize(hiddenSize);
        cache.blockMidAct[b].resize(hiddenSize);
        cache.blockConv2Out[b].resize(hiddenSize);

        Conv2dNCHW(cache.blockIn[b].data(), m_blocks[b].conv1, batch, height, width, cache.blockMidPre[b].data());
        for (size_t i = 0; i < hiddenSize; ++i) {
            cache.blockMidAct[b][i] = SiLU(cache.blockMidPre[b][i]);
        }

        Conv2dNCHW(cache.blockMidAct[b].data(), m_blocks[b].conv2, batch, height, width, cache.blockConv2Out[b].data());
        for (size_t i = 0; i < hiddenSize; ++i) {
            nextHidden[i] = cache.blockIn[b][i] + cache.blockConv2Out[b][i];
        }
        cur = nextHidden.data();
    }

    cache.headInput.assign(cur, cur + hiddenSize);
    cache.velocityPred.resize(static_cast<size_t>(batch) * m_arch.inputImageChannels * hw);
    Conv2dNCHW(cache.headInput.data(), m_head, batch, height, width, cache.velocityPred.data());
}

std::vector<float> FMBrain::DreamEulerODE(
    const float* initialNoise,
    uint32_t height,
    uint32_t width,
    uint32_t steps) const {
    const uint32_t stepCount = std::max(1u, steps);
    const size_t sampleSize = static_cast<size_t>(m_arch.inputImageChannels) * height * width;
    std::vector<float> x(initialNoise, initialNoise + sampleSize);
    const float dt = 1.0f / static_cast<float>(stepCount);

    ForwardCache cache;
    for (uint32_t s = 0; s < stepCount; ++s) {
        const float t = static_cast<float>(s) * dt; // same time label as the GPU dream (see DreamEulerODEGPU)
        Forward(x.data(), &t, 1, height, width, cache);
        for (size_t i = 0; i < sampleSize; ++i) {
            x[i] += dt * cache.velocityPred[i];
        }
    }
    return x;
}

DialScanReport FMBrain::ScanDials(float threshold) {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    ReadbackWeightsFromGPULocked();

    DialScanReport report;
    double sumAbs = 0.0;

    auto inspectVec = [&](const std::vector<float>& vec) {
        for (float w : vec) {
            const float mag = std::abs(w);
            report.totalDials++;
            if (mag > threshold) {
                report.busyDials++;
            } else {
                report.sleepingDials++;
            }
            if (mag > report.maxMagnitude) {
                report.maxMagnitude = mag;
            }
            sumAbs += mag;
        }
    };

    inspectVec(m_stem.weight);
    inspectVec(m_stem.bias);
    for (const auto& blk : m_blocks) {
        inspectVec(blk.conv1.weight);
        inspectVec(blk.conv1.bias);
        inspectVec(blk.conv2.weight);
        inspectVec(blk.conv2.bias);
    }
    inspectVec(m_head.weight);
    inspectVec(m_head.bias);

    if (report.totalDials > 0) {
        report.meanMagnitude = static_cast<float>(sumAbs / static_cast<double>(report.totalDials));
    }
    return report;
}

std::string FMBrain::ExportToJson() {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    ReadbackWeightsFromGPULocked();

    std::ostringstream oss;
    oss << "{\"preset\":" << static_cast<uint32_t>(m_arch.preset)
        << ",\"wave\":" << m_arch.waveLevel
        << ",\"width\":" << m_arch.width
        << ",\"blocks\":" << m_arch.numResBlocks
        << ",\"params\":" << ParameterCount()
        << ",\"weights\":[";

    bool first = true;
    auto emitVec = [&](const std::vector<float>& vec) {
        for (float v : vec) {
            if (!first) oss << ',';
            first = false;
            oss << v;
        }
    };
    emitVec(m_stem.weight);
    emitVec(m_stem.bias);
    for (const auto& blk : m_blocks) {
        emitVec(blk.conv1.weight);
        emitVec(blk.conv1.bias);
        emitVec(blk.conv2.weight);
        emitVec(blk.conv2.bias);
    }
    emitVec(m_head.weight);
    emitVec(m_head.bias);
    oss << "]}";
    return oss.str();
}

bool FMBrain::ImportFromJson(const std::string& json) {
    // 1. Parse and validate completely into temporary buffers BEFORE mutating existing brain!
    auto parseInt = [&](const char* key, uint32_t& out) -> bool {
        const size_t pos = json.find(key);
        if (pos == std::string::npos) return false;
        const size_t colon = json.find(':', pos);
        if (colon == std::string::npos) return false;
        try {
            out = static_cast<uint32_t>(std::stoul(json.substr(colon + 1)));
            return true;
        } catch (...) {
            return false;
        }
    };

    uint32_t presetRaw = 1;
    if (!parseInt("\"preset\"", presetRaw) || presetRaw > 2) {
        return false;
    }

    const size_t arrPos = json.find("\"weights\"");
    const size_t openBracket = (arrPos == std::string::npos) ? std::string::npos : json.find('[', arrPos);
    const size_t closeBracket = (openBracket == std::string::npos) ? std::string::npos : json.find(']', openBracket);
    if (openBracket == std::string::npos || closeBracket == std::string::npos) {
        return false;
    }

    // "wave" is optional: files written before wave brains existed have none and are pixel brains
    uint32_t waveRaw = 0;
    {
        const size_t wavePos = json.find("\"wave\"");
        if (wavePos != std::string::npos && wavePos < openBracket && !parseInt("\"wave\"", waveRaw)) return false;
        if (waveRaw > kMaxWaveLevel) return false;
    }

    const BrainPreset targetPreset = static_cast<BrainPreset>(presetRaw);
    const uint64_t expectedCount = ExpectedParameterCount(targetPreset, waveRaw);

    std::vector<float> parsed;
    parsed.reserve(static_cast<size_t>(expectedCount));
    const char* ptr = json.c_str() + openBracket + 1;
    const char* end = json.c_str() + closeBracket;
    while (ptr < end) {
        while (ptr < end && (*ptr == ' ' || *ptr == ',' || *ptr == '\n' || *ptr == '\r' || *ptr == '\t')) ++ptr;
        if (ptr >= end) break;
        char* next = nullptr;
        const float val = std::strtof(ptr, &next);
        if (next == ptr || !std::isfinite(val)) {
            return false;
        }
        parsed.push_back(val);
        ptr = next;
    }

    if (parsed.size() != expectedCount) {
        return false;
    }

    // 2. Validation passed: acquire lock, rebuild preset if needed, apply weights, and upload to GPU
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    InitializeFromScratchLocked(targetPreset, 0x1337BEEFULL, waveRaw);

    size_t offset = 0;
    auto loadVec = [&](std::vector<float>& vec) {
        std::copy_n(parsed.data() + offset, vec.size(), vec.data());
        offset += vec.size();
    };
    loadVec(m_stem.weight);
    loadVec(m_stem.bias);
    for (auto& blk : m_blocks) {
        loadVec(blk.conv1.weight);
        loadVec(blk.conv1.bias);
        loadVec(blk.conv2.weight);
        loadVec(blk.conv2.bias);
    }
    loadVec(m_head.weight);
    loadVec(m_head.bias);
    if (m_gpuReady) {
        UploadWeightsToGPULocked();
    }
    return true;
}

std::vector<uint8_t> FMBrain::ExportToBinary() {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    ReadbackWeightsFromGPULocked();

    const uint32_t magic = 0x4D46544Fu; // "OTFM"
    // the second byte of the preset word holds the wave level (older files have zero there: pixel brains)
    const uint32_t presetVal = static_cast<uint32_t>(m_arch.preset) | (m_arch.waveLevel << 8);
    const uint64_t count = ParameterCount();

    std::vector<uint8_t> out(16 + count * sizeof(float));
    std::memcpy(out.data() + 0, &magic, 4);
    std::memcpy(out.data() + 4, &presetVal, 4);
    std::memcpy(out.data() + 8, &count, 8);

    size_t byteOffset = 16;
    auto appendVec = [&](const std::vector<float>& vec) {
        const size_t bytes = vec.size() * sizeof(float);
        std::memcpy(out.data() + byteOffset, vec.data(), bytes);
        byteOffset += bytes;
    };
    appendVec(m_stem.weight);
    appendVec(m_stem.bias);
    for (const auto& blk : m_blocks) {
        appendVec(blk.conv1.weight);
        appendVec(blk.conv1.bias);
        appendVec(blk.conv2.weight);
        appendVec(blk.conv2.bias);
    }
    appendVec(m_head.weight);
    appendVec(m_head.bias);
    return out;
}

bool FMBrain::ImportFromBinary(const std::vector<uint8_t>& bytes) {
    // 1. Validate header, expected parameter count, and payload length BEFORE mutating active brain!
    if (bytes.size() < 16) return false;
    uint32_t magic = 0;
    uint32_t presetVal = 0;
    uint64_t count = 0;
    std::memcpy(&magic, bytes.data() + 0, 4);
    std::memcpy(&presetVal, bytes.data() + 4, 4);
    std::memcpy(&count, bytes.data() + 8, 8);
    if (magic != 0x4D46544Fu || (presetVal & 0xFFu) > 2 || (presetVal >> 8) > kMaxWaveLevel) return false;

    const BrainPreset targetPreset = static_cast<BrainPreset>(presetVal & 0xFFu);
    const uint32_t waveRaw = presetVal >> 8;
    const uint64_t expectedCount = ExpectedParameterCount(targetPreset, waveRaw);
    if (count != expectedCount || bytes.size() != 16 + count * sizeof(float)) {
        return false;
    }

    std::vector<float> parsed(static_cast<size_t>(count));
    std::memcpy(parsed.data(), bytes.data() + 16, static_cast<size_t>(count) * sizeof(float));
    for (float v : parsed) {
        if (!std::isfinite(v)) return false;
    }

    // 2. Validation passed: commit atomically under mutex
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    InitializeFromScratchLocked(targetPreset, 0x1337BEEFULL, waveRaw);

    size_t offset = 0;
    auto readVec = [&](std::vector<float>& vec) {
        std::copy_n(parsed.data() + offset, vec.size(), vec.data());
        offset += vec.size();
    };
    readVec(m_stem.weight);
    readVec(m_stem.bias);
    for (auto& blk : m_blocks) {
        readVec(blk.conv1.weight);
        readVec(blk.conv1.bias);
        readVec(blk.conv2.weight);
        readVec(blk.conv2.bias);
    }
    readVec(m_head.weight);
    readVec(m_head.bias);
    if (m_gpuReady) {
        UploadWeightsToGPULocked();
    }
    return true;
}

} // namespace XboxOTFM
