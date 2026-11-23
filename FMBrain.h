#pragma once

#include <d3d12.h>
#include <DirectML.h>
#include <wrl/client.h>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace XboxOTFM {

enum class BrainPreset : uint32_t {
    Low25k = 0,    // Width 26, 2 residual blocks (~25k dials)
    Medium80k = 1, // Width 38, 3 residual blocks (~80k dials)
    Max250k = 2    // Width 58, 4 residual blocks (~250k dials)
};

enum class ComputeEngine : uint32_t {
    GPU = 0,
    CPU = 1
};

struct BrainArchitecture {
    BrainPreset preset = BrainPreset::Medium80k;
    // 0 = the brain works on pixels. 1 = it works on one Haar "wave" step of the picture: 3 colours x 4 bands = 12 channels at half the
    // width and height (about 4 times less work per lesson; nothing is lost). Chosen when the brain is made, never changed afterwards.
    uint32_t waveLevel = 0;
    uint32_t inputImageChannels = 3;  // 3 for pixels, 12 for wave form
    uint32_t coordChannels = 2;       // (x, y) in [-1, 1]
    uint32_t harmonicBands = 4;       // k = 0..3 -> sin/cos for x and y = 16 channels
    uint32_t timeChannels = 4;        // sinusoidal embedding of scalar t
    uint32_t width = 38;
    uint32_t numResBlocks = 3;

    uint32_t TotalInputChannels() const noexcept {
        return inputImageChannels + coordChannels + (harmonicBands * 4) + timeChannels;
    }
};

struct DialScanReport {
    uint64_t totalDials = 0;
    uint64_t busyDials = 0;     // |weight| > 0.01f
    uint64_t sleepingDials = 0; // |weight| <= 0.01f
    float maxMagnitude = 0.0f;
    float meanMagnitude = 0.0f;
};

struct ConvLayerWeights {
    uint32_t inChannels = 0;
    uint32_t outChannels = 0;
    uint32_t kernelSize = 3; // 1 or 3
    std::vector<float> weight; // [outC, inC, K, K]
    std::vector<float> bias;   // [outC]
};

struct ResidualBlockWeights {
    ConvLayerWeights conv1; // 3x3
    ConvLayerWeights conv2; // 3x3
    std::vector<float> mW1, vW1, mB1, vB1;
    std::vector<float> mW2, vW2, mB2, vB2;
};

struct ForwardCache {
    uint32_t batch = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    std::vector<float> stemInput;              // [N, Cin, H, W]
    std::vector<float> stemPreAct;             // [N, W, H, W]
    std::vector<float> stemOut;                // [N, W, H, W]
    std::vector<std::vector<float>> blockIn;   // [B][N, W, H, W]
    std::vector<std::vector<float>> blockMidPre;
    std::vector<std::vector<float>> blockMidAct;
    std::vector<std::vector<float>> blockConv2Out;
    std::vector<float> headInput;              // [N, W, H, W]
    std::vector<float> velocityPred;           // [N, C, H, W] (C = 3 for pixels, 12 for wave form)
};

// ---------------------------------------------------------------------------------------------
// GPU support types (private to the compute implementation, public only so they can be unit-tested)
// ---------------------------------------------------------------------------------------------

// 24 DWORDs. Must match `cbuffer TrainConstants` in FMCompute.hlsl field for field.
struct GPUTrainConstants {
    uint32_t gBatch;
    uint32_t gWidth;
    uint32_t gHeight;
    uint32_t gInC;
    uint32_t gOutC;
    uint32_t gKernelSize;
    uint32_t gApplySiLU;
    uint32_t gAddResidual;
    uint32_t gSiLUGradEpi;
    uint32_t gUseConstT;
    uint32_t gParamCount;
    uint32_t gSampleStride;
    uint32_t gPartialCount;
    uint32_t gPartialOffset;
    float gTimeT;
    float gDt;
    float gLearningRate;
    float gBeta1;
    float gBeta2;
    float gEpsilon;
    float gBiasCorr1;
    float gBiasCorr2;
    float gInvOutCount;
    float gGradClipNorm;
};
static_assert(sizeof(GPUTrainConstants) == 24 * sizeof(uint32_t),
              "GPUTrainConstants must be exactly 24 DWORDs (it is sent as root constants)");

// A D3D12 buffer plus the resource state this class believes it is in.
struct GpuBuffer {
    Microsoft::WRL::ComPtr<ID3D12Resource> res;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    uint64_t bytes = 0;
    bool fixedState = false; // UPLOAD / READBACK heaps never change state
    bool uavDirty = false;   // written by a compute shader since the last UAV barrier
};

// Order must match kKernelFiles in FMBrain.cpp.
enum class GpuKernel : uint32_t {
    EncodeSine = 0,
    ConvFwd,
    MseGrad,
    ConvBwdIn,
    SiLUBwd,
    ConvBwdW,
    SumSquares,
    ReduceFinal,
    Adam,
    Euler,
    Count
};
constexpr uint32_t kGpuKernelCount = static_cast<uint32_t>(GpuKernel::Count);

// Where one conv layer lives inside the flat parameter arenas (offsets are in floats, 64-float aligned).
struct GpuLayerInfo {
    uint32_t inC = 0;
    uint32_t outC = 0;
    uint32_t k = 1;
    uint64_t wOff = 0;
    uint64_t bOff = 0;
    uint64_t wCount = 0;
    uint64_t bCount = 0;
};

struct GpuBinding {
    GpuBuffer* buf = nullptr;
    uint64_t byteOffset = 0;
};

struct GpuDispatch {
    GpuKernel kernel = GpuKernel::EncodeSine;
    GPUTrainConstants cb = {};
    GpuBinding srv[4];
    GpuBinding uav[3];
    uint32_t gx = 1;
    uint32_t gy = 1;
    uint32_t gz = 1;
    double work = 0.0; // approximate multiply-adds, used to split long steps into several submissions
};

struct GpuActivations {
    uint64_t capacityPixels = 0; // batch * H * W these buffers can hold
    GpuBuffer xt, vTarget, tValues, stemIn, stemPre, vPred, gradPred, gradA, gradB, gradTmp;
    std::vector<GpuBuffer> act;    // numBlocks + 1: act[0] = stem output ... act[B] = head input
    std::vector<GpuBuffer> midPre; // [numBlocks] pre-activation of conv1
    std::vector<GpuBuffer> midAct; // [numBlocks] SiLU(conv1)
    GpuBuffer upload;              // CPU -> GPU staging (UPLOAD heap)
    GpuBuffer readback;            // GPU -> CPU staging (READBACK heap)
};

struct FMBrainTestAccess; // lets the offline test harness look inside; unused in the shipping app

class FMBrain {
public:
    explicit FMBrain(BrainPreset preset = BrainPreset::Medium80k, uint32_t waveLevel = 0);
    ~FMBrain();

    // Rebuilds all layers from 0% pre-trained weights (pure random Kaiming/He noise). waveLevel: 0 = pixel brain, 1 = wave brain.
    void InitializeFromScratch(BrainPreset preset, uint64_t seed = 0x1337BEEFULL, uint32_t waveLevel = 0);

    // Creates the compute queue, root signature, pipelines (from Shaders\*.cso) and GPU-resident weight /
    // Adam-moment / activation buffers, then PROVES the GPU math against the CPU path with a quick self-test.
    // Returns a failure HRESULT (and writes the exact reason to crash_log.txt) if anything is wrong; the
    // caller then keeps training on the CPU.
    HRESULT InitializeGPU(ID3D12Device* d3dDevice, ID3D12CommandQueue* cmdQueue);
    void ReleaseGPU();

    // Uploads CPU weights + Adam moments -> GPU-resident buffers (only on init/import/engine switch, never per-lesson).
    HRESULT UploadWeightsToGPU();
    // Syncs GPU-resident weights -> CPU vectors (only on Scan Dials / Export Brain / Switch to CPU).
    HRESULT ReadbackWeightsFromGPU();

    // Tells the brain the CPU vectors were changed by the CPU trainer, so the GPU copy must be refreshed
    // before it is used again.
    void NotifyCpuWeightsChanged();

    // Runs an entire forward + MSE grad + backward + gradient clipping + Adam update purely on the GPU
    // without per-step CPU weight copies. `batch` may be reduced internally (memory / 128x128 safety).
    HRESULT TrainStepGPU(
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
        float gradClipNorm = 1.0f);

    // Whole Euler trajectory t=0 -> 1 (1, 2 or 6 steps) on the GPU: the intermediate images never leave the GPU,
    // only the final [C, H, W] sample (C = 3 for a pixel brain, 12 for a wave brain) is copied back once.
    HRESULT DreamEulerODEGPU(
        const float* initialNoise,
        uint32_t height,
        uint32_t width,
        uint32_t steps,
        std::vector<float>& outImageCHW);

    // Largest mini-batch the GPU activation budget allows at this network size (the width/height the brain really works at: for a wave
    // brain that is half the picture size) for the current preset.
    uint32_t MaxGpuBatchForResolution(uint32_t resolution) const noexcept;

    const BrainArchitecture& Architecture() const noexcept { return m_arch; }
    uint32_t WaveLevel() const noexcept { return m_arch.waveLevel; }
    uint64_t ParameterCount() const noexcept;
    static constexpr uint32_t kMaxWaveLevel = 1;
    // Image channels the brain works with: 3 for pixels, 12 for the wave form (waveLevel is limited to 0..kMaxWaveLevel).
    static uint32_t ImageChannelsForWaveLevel(uint32_t waveLevel) noexcept { return waveLevel == 0 ? 3u : 12u; }
    static uint64_t ExpectedParameterCount(BrainPreset preset, uint32_t waveLevel = 0) noexcept;
    bool IsGPUReady() const noexcept { return m_gpuReady; }
    bool IsDirectMLReady() const noexcept { return m_dmlReady; }

    // Encodes x_t [N, C, H, W] (C = 3 pixel brain / 12 wave brain) + [-1,1] spatial grid + sin/cos(2^k * pi * xy) k=0..3 + time embedding t.
    void BuildSineOperatorInput(
        const float* xt,
        const float* tValues,
        uint32_t batch,
        uint32_t height,
        uint32_t width,
        std::vector<float>& outEncoded) const;

    // CPU Forward pass producing predicted velocity field v_pred [N, C, H, W] (C as above).
    void Forward(
        const float* xt,
        const float* tValues,
        uint32_t batch,
        uint32_t height,
        uint32_t width,
        ForwardCache& cache) const;

    // CPU Euler ODE integration from t=0 -> t=1 in 1, 2, or 6 steps (the GPU twin is DreamEulerODEGPU).
    std::vector<float> DreamEulerODE(
        const float* initialNoise,
        uint32_t height,
        uint32_t width,
        uint32_t steps) const;

    // Scans all model dials counting busy (|w| > 0.01) vs sleeping weights.
    DialScanReport ScanDials(float threshold = 0.01f);

    // Serialization for Phone LAN Bridge export/import (validates before mutating brain).
    std::string ExportToJson();
    bool ImportFromJson(const std::string& json);
    std::vector<uint8_t> ExportToBinary();
    bool ImportFromBinary(const std::vector<uint8_t>& bytes);

    std::recursive_mutex& Mutex() noexcept { return m_brainMutex; }

    // Direct accessors for CPU Trainer backpropagation & Adam updates.
    ConvLayerWeights& Stem() noexcept { return m_stem; }
    std::vector<float>& StemMW() noexcept { return m_stemMW; }
    std::vector<float>& StemVW() noexcept { return m_stemVW; }
    std::vector<float>& StemMB() noexcept { return m_stemMB; }
    std::vector<float>& StemVB() noexcept { return m_stemVB; }

    std::vector<ResidualBlockWeights>& Blocks() noexcept { return m_blocks; }

    ConvLayerWeights& Head() noexcept { return m_head; }
    std::vector<float>& HeadMW() noexcept { return m_headMW; }
    std::vector<float>& HeadVW() noexcept { return m_headVW; }
    std::vector<float>& HeadMB() noexcept { return m_headMB; }
    std::vector<float>& HeadVB() noexcept { return m_headVB; }

private:
    friend struct FMBrainTestAccess;

    static BrainArchitecture PresetConfig(BrainPreset preset, uint32_t waveLevel);
    static void InitConvLayer(ConvLayerWeights& layer, uint32_t inC, uint32_t outC, uint32_t k, uint64_t& rngState);

    void InitializeFromScratchLocked(BrainPreset preset, uint64_t seed, uint32_t waveLevel);
    void ReleaseGPULocked();
    void ParkGpuObjectsLocked();                 // hung GPU: keep what it may still touch alive instead of freeing it
    static void ReleaseParkedGpuObjectsForTests(); // test harness only
    HRESULT UploadWeightsToGPULocked();
    HRESULT ReadbackWeightsFromGPULocked();

    // ---- GPU resources ----
    HRESULT CreateGpuBuffer(uint64_t byteSize, D3D12_HEAP_TYPE heapType, GpuBuffer& out);
    bool GpuLayoutMatchesLocked() const noexcept;
    HRESULT RebuildGpuLayoutLocked();
    HRESULT EnsureActivationBuffersLocked(uint64_t neededPixels);
    HRESULT EnsureGpuInSyncLocked();
    HRESULT ZeroGradArenaLocked();
    uint64_t ActivationBytesPerPixel() const noexcept;

    // ---- queue / command recording ----
    HRESULT WaitComputeQueue();
    HRESULT GpuBegin();
    HRESULT GpuFlush(bool reopen);
    HRESULT GpuFinishAndWait();
    void GpuAbort();
    void TransitionTo(GpuBuffer& buf, D3D12_RESOURCE_STATES state);
    HRESULT RecordDispatch(const GpuDispatch& d);
    void RecordUploadCopy(GpuBuffer& dst, uint64_t dstOffset, uint64_t uploadOffset, uint64_t bytes);
    void RecordReadbackCopy(GpuBuffer& src, uint64_t srcOffset, uint64_t readbackOffset, uint64_t bytes);
    HRESULT NoteGpuFailure(const char* stage, HRESULT hr);

    // ---- recorded passes ----
    HRESULT RecordForwardPass(uint32_t batch, uint32_t height, uint32_t width, bool useConstT, float constT);
    HRESULT RecordBackwardPass(uint32_t batch, uint32_t height, uint32_t width);
    HRESULT RecordOptimizerPass(uint32_t batch, uint32_t height, uint32_t width, uint64_t stepIndex,
                                float lr, float beta1, float beta2, float eps, float gradClipNorm);
    HRESULT RunForwardGpuLocked(const float* xt, const float* tValues, uint32_t batch, uint32_t height,
                                uint32_t width, std::vector<float>& outVelocity);
    HRESULT TrainStepGpuLocked(const float* xt, const float* vTarget, const float* tValues, uint32_t batch,
                               uint32_t height, uint32_t width, uint64_t stepIndex, float lr, float beta1,
                               float beta2, float eps, float gradClipNorm, float* outStats);
    HRESULT SelfTestGpuLocked(std::string& report);

    mutable std::recursive_mutex m_brainMutex;
    BrainArchitecture m_arch;

    ConvLayerWeights m_stem;
    std::vector<float> m_stemMW, m_stemVW, m_stemMB, m_stemVB;

    std::vector<ResidualBlockWeights> m_blocks;

    ConvLayerWeights m_head;
    std::vector<float> m_headMW, m_headVW, m_headMB, m_headVB;

    // DirectML (optional probe) & D3D12 GPU-resident training resources
    bool m_dmlReady = false;
    bool m_gpuReady = false;
    bool m_gpuFaulted = false;             // device removed / GPU hung: the GPU path stays off until re-initialised
    bool m_gpuWeightsDirtyOnDevice = false; // GPU weights are newer than the CPU vectors
    bool m_gpuNeedsUpload = true;           // CPU vectors are newer than (or different from) the GPU copy
    bool m_gpuLayoutValid = false;
    bool m_recording = false;
    bool m_inFlight = false;
    bool m_loggedFirstStep = false;
    uint32_t m_loggedBatchClampRes = 0;
    uint32_t m_verifiedLayouts = 0;         // bit per wave level whose GPU maths passed the self-test during this GPU session
    double m_pendingWork = 0.0;
    uint32_t m_failureLogCount = 0;
    std::chrono::steady_clock::time_point m_lastWeightSync{};

    Microsoft::WRL::ComPtr<ID3D12Device> m_d3dDevice;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_cmdQueue;      // the app's graphics queue (not used for compute work)
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_computeQueue;  // dedicated compute queue (or the graphics queue as fallback)
    D3D12_COMMAND_LIST_TYPE m_listType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_computeAlloc;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_computeCmdList;
    Microsoft::WRL::ComPtr<ID3D12Fence> m_computeFence;
    HANDLE m_computeFenceEvent = nullptr;
    uint64_t m_computeFenceValue = 0;

    Microsoft::WRL::ComPtr<IDMLDevice> m_dmlDevice;
    Microsoft::WRL::ComPtr<IDMLCommandRecorder> m_dmlRecorder;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSig;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pso[kGpuKernelCount];

    // Flat parameter arenas: every layer's weights then bias, each 64-float aligned.
    BrainArchitecture m_gpuArch;
    std::vector<GpuLayerInfo> m_gpuLayers; // stem, (conv1, conv2) x blocks, head
    uint64_t m_arenaFloats = 0;
    uint64_t m_arenaBytes = 0;
    GpuBuffer m_gpuParams, m_gpuGrads, m_gpuAdamM, m_gpuAdamV;
    GpuBuffer m_gpuPartials; // 2 * 4096 floats: gradient-norm partials then loss partials
    GpuBuffer m_gpuStats;    // [0] ||g||^2, [1] Adam gradient scale, [2] MSE loss, [3] ||g||

    GpuActivations m_act;
};

} // namespace XboxOTFM
