"""Build VAE executor probes without modifying the pinned MNN checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def once(body, old, new):
    if body.count(old) != 1:
        raise ValueError('Pinned MNN source anchor has changed: ' + old.splitlines()[0])
    return body.replace(old, new)


def instrument(source):
    source = once(source, '        ::memset(tempOutPtr, 0, outputSize);', '''        auto* trace = LevoDeconvTrace::current.load();
        auto& worker = trace->workers[tId];
        if (tId == 0) {
            trace->threadCount = threadNumber;
            trace->elementBytes = core->bytes;
            trace->packWidth = core->pack;
            trace->tile = eP;
            trace->kernel = kw * kh;
            trace->stride = strideX * strideY;
            trace->dilation = dilateX * dilateY;
            trace->outputBytes = outputSize;
            trace->packBytesPerWorker = im2colOutputStride;
            trace->gemmBytesPerWorker = gemmOutputStride;
        }
        auto tick = LevoDeconvTrace::Clock::now();
        ::memset(tempOutPtr, 0, outputSize);
        LevoDeconvTrace::mark(worker, LevoDeconvTrace::Clear, tick);''')
    source = once(source, '''            core->MNNPackC4ForMatMul_A((float*)(gemmInputBufferPtr), (const float**)(&aStart), info, stride);
            if (xCount == eP) {''', '''            tick = LevoDeconvTrace::Clock::now();
            core->MNNPackC4ForMatMul_A((float*)(gemmInputBufferPtr), (const float**)(&aStart), info, stride);
            LevoDeconvTrace::mark(worker, LevoDeconvTrace::Pack, tick);
            if (xCount == eP) {''')
    source = once(source, '            // Col2Im', '''            LevoDeconvTrace::mark(worker, LevoDeconvTrace::Gemm, tick);
            // Col2Im''')
    source = once(source, '''        }
    }, threadNumber);
    auto second = std::make_pair''', '''            LevoDeconvTrace::mark(worker, LevoDeconvTrace::Scatter, tick);
        }
    }, threadNumber);
    auto second = std::make_pair''')
    source = once(source, '        auto biasPtr = biasTensor->host<uint8_t>();', '''        auto biasPtr = biasTensor->host<uint8_t>();
        auto& worker = LevoDeconvTrace::current.load()->workers[tId];''')
    source = once(source, '''            if (threadNumber > 1) {
                for (int index=0; index<threadNumber-1; ++index) {''', '''            auto tick = LevoDeconvTrace::Clock::now();
            if (threadNumber > 1) {
                for (int index=0; index<threadNumber-1; ++index) {''')
    needle = '            core->MNNAxByClampBroadcastUnit((float*)dstZ, (float*)dstZ, (const float*)((uint8_t*)biasPtr +  unitBytes * z), src_height * src_width * batch, 0, 0, 1, mPostParameters.data());'
    source = once(source, needle, '''            LevoDeconvTrace::mark(worker, LevoDeconvTrace::Reduce, tick);
'''+needle+'''
            LevoDeconvTrace::mark(worker, LevoDeconvTrace::Bias, tick);''')
    start = '''    for (auto& unit : mExecuteFuntion) {
        MNN_CONCURRENCY_BEGIN(tId, unit.second) {'''
    source = once(source, start, '''    LevoDeconvTrace::Trace trace;
    LevoDeconvTrace::current.store(&trace);
    std::array<uint64_t, 2> wall{};
    int phase = 0;
    for (auto& unit : mExecuteFuntion) {
        if (unit.second > static_cast<int>(trace.workers.size())) {
            LevoDeconvTrace::current.store(nullptr);
            return NOT_SUPPORT;
        }
        auto phaseStart = LevoDeconvTrace::Clock::now();
        MNN_CONCURRENCY_BEGIN(tId, unit.second) {''')
    source = once(source, '''        MNN_CONCURRENCY_END();
    }
    return NO_ERROR;
}
class CPUDeconvolutionCreator''', '''        MNN_CONCURRENCY_END();
        wall.at(phase++) = LevoDeconvTrace::elapsed(phaseStart);
    }
    LevoDeconvTrace::current.store(nullptr);
    LevoDeconvTrace::emit(trace, inputs[0]->channel(), outputs[0]->channel(),
                          inputs[0]->elementSize(), outputs[0]->elementSize(), wall[0], wall[1]);
    return NO_ERROR;
}
class CPUDeconvolutionCreator''')
    return '#include "VaeDeconvolutionTrace.hpp"\n' + source


def instrument_epilogue(source):
    """Time dispatch phases on both routes and enable the alternate epilogue by environment."""
    second = '    auto second = std::make_pair([ocC4, src_height, src_width, threadNumber, batch, biasTensor, this, outputSize, core]'
    source = once(source, second, '''    const auto* switchValue = std::getenv("LEVO_VAE_REDUCE_BIAS");
    const bool fusedEpilogue = switchValue && std::strcmp(switchValue, "1") == 0 &&
        core->bytes == 2 && core->pack == 8 && threadNumber == 4 && mPostParameters[1] == 1.0f;
    std::printf("LEVO_VAE_EPILOGUE_SETUP %d %d %d %zu %d\\n", output->channel(), src_width * src_height * batch,
                threadNumber, static_cast<std::size_t>(outputSize), int(fusedEpilogue));
''' + second[:-1] + ', fusedEpilogue]')
    anchor = '''            auto dstZ = outputPtr + z * src_height * src_width * batch * unitBytes;
            if (threadNumber > 1) {'''
    source = once(source, anchor, '''            auto dstZ = outputPtr + z * src_height * src_width * batch * unitBytes;
            if (fusedEpilogue) {
                const std::size_t positions = std::size_t(src_height) * src_width * batch;
                const auto* partials = mExtraOutput.ptr() + z * positions * unitBytes;
                LevoDeconvEpilogue::run(dstZ, partials, outputSize, biasPtr + unitBytes * z,
                                      positions, mPostParameters.data());
                continue;
            }
            if (threadNumber > 1) {''')
    start = '''    for (auto& unit : mExecuteFuntion) {
        MNN_CONCURRENCY_BEGIN(tId, unit.second) {'''
    source = once(source, start, '''    std::array<uint64_t, 2> duration{};
    std::size_t phase = 0;
    for (auto& unit : mExecuteFuntion) {
        const auto phaseStart = std::chrono::steady_clock::now();
        MNN_CONCURRENCY_BEGIN(tId, unit.second) {''')
    end = '''        MNN_CONCURRENCY_END();
    }
    return NO_ERROR;
}
class CPUDeconvolutionCreator'''
    source = once(source, end, '''        MNN_CONCURRENCY_END();
        duration.at(phase++) = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - phaseStart).count();
    }
    std::printf("LEVO_VAE_EPILOGUE_PHASES %d %d %llu %llu\\n", outputs[0]->channel(),
                outputs[0]->elementSize() / outputs[0]->channel(),
                static_cast<unsigned long long>(duration[0]), static_cast<unsigned long long>(duration[1]));
    return NO_ERROR;
}
class CPUDeconvolutionCreator''')
    return ('#include "VaeDeconvolutionEpilogue.hpp"\n#include <array>\n#include <chrono>\n'
            '#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n' + source)


def instrument_convolution(source):
    symbol = 'ErrorCode DenseConvInt8TiledExecutor::onExecute('
    prefix, body = source.split(symbol)
    a = body.index('    auto tileSplitFunction =')
    b = body.index('    auto ocSplitFunction =')
    c = body.index('        MNN_CONCURRENCY_BEGIN(tId, threads)', b)
    d = body.index('    if (!mSplitByOc)', c)
    chunks = [body[:a], body[a:b], body[b:c], body[c:d], body[d:]]
    signatures = [
        (r'\bBatch(?:Asy|Sym)DynamicQuant\(', 'Quant', 9),
        (r'(?:\bblitProc|mRelatedFunctions\.MNNGeneralIm2Col)\(', 'Pack', 4),
        (r'\b(?:mGemmKernel|gemmInt8)\(', 'GemmEpilogue', 2),
        (r'mRelatedFunctions\.MNNSumByAxisLForMatmul_A\(', 'InputSum', 2),
        (r'\bmQuantFunc\(\(float\*\)mTempOutput\.ptr\(\)', 'OutputQuant', 1),
        (r'\b_onlineReorderWeight(?:Pack|KernelSum)H(?:128|8)ToH32\(', 'WeightReorder', 4),
    ]
    counts = dict.fromkeys((name for _, name, _ in signatures), 0)
    for index, (chunk, worker) in enumerate(zip(chunks, ('threads', 'tId', 'threads', 'tId', 'threads'), strict=True)):
        edits = []
        for pattern, stage, _ in signatures:
            for match in re.finditer(pattern, chunk):
                begin = match.start()
                end = chunk.index('(', begin) + 1
                depth = 1
                while depth:
                    depth += (chunk[end] == '(') - (chunk[end] == ')')
                    end += 1
                if chunk[end] != ';':
                    raise ValueError('Quantized convolution probe must wrap an independent call')
                edits.append((begin, end + 1, f'levoTrace.measure({worker}, LevoConvTrace::{stage}, '
                              f'[&] {{ {chunk[begin:end+1]} }});'))
                counts[stage] += 1
        for begin, end, patch in sorted(edits, reverse=True):
            chunk = chunk[:begin] + patch + chunk[end:]
        chunks[index] = chunk
    if counts != {name: count for _, name, count in signatures}:
        raise ValueError('Pinned quantized convolution anchors changed')
    needle = '    const int threads = static_cast<CPUBackend*>(backend())->threadNumber();'
    body = once(''.join(chunks), needle, needle + '''
    LevoConvTrace::Trace levoTrace(threads, plane, ic, oc, mResourceInt8->mWeightBits,
        mResourceInt8->mDynamicQuant, dynamicOption, mSplitByOc, mIm2ColBasedInt8,
        DST_XUNIT, SRC_UNIT, UNIT, gcore->bytes, dstBytes, kxky,
        mIm2ColParamter.strideX * mIm2ColParamter.strideY);''')
    return '#include "VaeConvolutionTrace.hpp"\n' + prefix + symbol + body


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mnn_root', 'mnn_build', 'native_build', 'ninja', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--with-convolution', action='store_true')
    parser.add_argument('--epilogue-comparison', action='store_true',
                        help='Compare original and single-pass FP16 Reduce/Bias with phase-only timers')
    parser.add_argument('--layout-alias', action='store_true', help='Link the existing layout alias probe')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    mnn = args.mnn_root.resolve()
    root = Path(__file__).resolve().parents[1]
    binary = output / 'vae-executor-profile'

    def command(build, target):
        return subprocess.check_output([str(args.ninja), '-C', str(build), '-t', 'commands', target],
                                       text=True).strip().splitlines()[-1]

    deconvolution_probe = instrument_epilogue if args.epilogue_comparison else instrument
    recipes = [('source/backend/cpu/CPUDeconvolution.cpp', deconvolution_probe)]
    if args.with_convolution:
        recipes.append(('source/backend/cpu/compute/ConvInt8TiledExecutor.cpp', instrument_convolution))
    objects, compile_lines, hashes = [], [], {}
    for path, inject in recipes:
        source = mnn / path
        rewritten = output / f'instrumented-{source.name}'
        rewritten.write_text(inject(source.read_text(encoding='utf-8')), encoding='utf-8')
        obj = rewritten.with_suffix('.o')
        line = command(args.mnn_build, f'mnn/CMakeFiles/MNNCPU.dir/{path}.o')
        if not line.endswith(source.as_posix()) or ' -MD -MT ' not in line:
            raise ValueError('Unexpected pinned MNN compile command')
        fp16_flag = ' -march=armv8.2-a+fp16' if inject is instrument_epilogue else ''
        compile_lines.append(line.split(' -MD -MT ', 1)[0] + fp16_flag
                             + f' -I"{source.parent}" -I"{root / "native-bench"}" '
                             f'-o "{obj}" -c "{rewritten}"')
        objects.append(f'"{obj}"')
        hashes[source.name] = digest(source)
    target = 'levo_stream_vae_alias' if args.layout_alias else 'levo_stream_vae_fusion'
    link_line = command(args.native_build, target)
    prefix, suffix, anchor = ' /C "cd . && ', ' && cd ."', f' -o {target} '
    if prefix not in link_line or not link_line.endswith(suffix) or link_line.count(anchor) != 1:
        raise ValueError('Unexpected isolated VAE link command')
    link_line = link_line.split(prefix, 1)[1].removesuffix(suffix).replace(anchor, f' {" ".join(objects)} -o "{binary}" ')
    with (output / 'build.log').open('w', encoding='utf-8') as log:
        for compile_line in compile_lines:
            subprocess.run(compile_line, cwd=args.mnn_build, check=True, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run(link_line, cwd=args.native_build, check=True, stdout=log, stderr=subprocess.STDOUT)
    (output / 'build.json').write_text(json.dumps({'sourceSha256': hashes,
                                                   'binarySha256': digest(binary)}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
