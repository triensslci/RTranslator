/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copyFile of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package nie.translator.rtranslator.voice_translation.neural_networks.voice;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.SystemClock;
import android.util.Log;

import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.extensions.OrtxPackage;
import nie.translator.rtranslator.BuildConfig;
import nie.translator.rtranslator.Global;
import nie.translator.rtranslator.R;
import nie.translator.rtranslator.tools.CustomLocale;
import nie.translator.rtranslator.tools.ErrorCodes;
import nie.translator.rtranslator.tools.nn.CacheContainerNative;
import nie.translator.rtranslator.tools.nn.TensorUtils;
import nie.translator.rtranslator.tools.nn.Utils;
import nie.translator.rtranslator.voice_translation.neural_networks.NeuralNetworkApi;


public class Recognizer extends NeuralNetworkApi {
    private static final int MAX_TOKENS_PER_SECOND = 30;
    private static final int MAX_TOKENS = 445;   //if we generate more than this quantity of tokens for a transcription we have an error
    private static final double NO_SPEECH_THRESHOLD = 0.6;
    private static final double NO_SPEECH_LOW_CONFIDENCE_THRESHOLD = -1.0;
    private static final int TRIM_WINDOW_MILLIS = 100;
    private static final int TRIM_PADDING_BEFORE_MILLIS = 300;
    private static final int TRIM_PADDING_AFTER_MILLIS = 500;
    private static final float TRIM_PEAK_THRESHOLD = 900f / 32767f;
    private static final float TRIM_RMS_THRESHOLD = 180f / 32767f;
    private static final float NORMALIZATION_TARGET_RMS = 0.08f;
    private static final float NORMALIZATION_MAX_GAIN = 4f;
    private static final float NORMALIZATION_PEAK_LIMIT = 0.95f;
    private static final float NORMALIZATION_MIN_RMS = 0.001f;
    private static final double AUDIO_LANGUAGE_MIN_PROBABILITY = 0.75;
    private static final double AUDIO_LANGUAGE_PROBABILITY_MARGIN = 0.20;
    public static final String UNDEFINED_TEXT = "[(und)]";
    private ArrayList<RecognizerListener> callbacks = new ArrayList<>();
    private ArrayList<RecognizerMultiListener> multiCallbacks = new ArrayList<>();
    private boolean recognizing = false;
    private ArrayDeque<DataContainer> dataToRecognize = new ArrayDeque<>();
    private final Object lock = new Object();

    private static final int BLANK_TOKEN_ID = 220;
    private static final int EOS_TOKEN_ID = 50257;
    private static final int START_TOKEN_ID = 50258;
    private static final int TRANSLATE_TOKEN_ID = 50358;
    private static final int TRANSCRIBE_TOKEN_ID = 50359;
    private static final int START_OF_LM_TOKEN_ID = 50360;
    private static final int START_OF_PREV_TOKEN_ID = 50361;
    private static final int NO_SPEECH_TOKEN_ID = 50362;
    private static final int NO_TIMESTAMPS_TOKEN_ID = 50363;
    private static final int FIRST_TIMESTAMP_TOKEN_ID = 50364;
    private static final int WHISPER_LAYER_COUNT = 12;
    private static final int WHISPER_HEAD_COUNT = 12;
    private static final int WHISPER_HEAD_SIZE = 64;
    private static final String PHOWHISPER_ENCODER_MODEL = "PhoWhisper_encoder.onnx";
    private static final String PHOWHISPER_DECODER_INIT_MODEL = "PhoWhisper_decoder_init.onnx";
    private static final String PHOWHISPER_DECODER_MODEL = "PhoWhisper_decoder.onnx";
    private static final String LEGACY_WHISPER_ENCODER_MODEL = "Whisper_encoder.onnx";
    private static final String LEGACY_WHISPER_CACHE_INIT_MODEL = "Whisper_cache_initializer.onnx";
    private static final String LEGACY_WHISPER_CACHE_INIT_BATCH_MODEL = "Whisper_cache_initializer_batch.onnx";
    private static final String LEGACY_WHISPER_DECODER_MODEL = "Whisper_decoder.onnx";
    private static final OrtSession.SessionOptions.OptLevel WHISPER_GRAPH_OPT_LEVEL = OrtSession.SessionOptions.OptLevel.NO_OPT;

    private static final String[] LANGUAGES = {
            "en",
            "zh",
            "de",
            "es",
            "ru",
            "ko",
            "fr",
            "ja",
            "pt",
            "tr",
            "pl",
            "ca",
            "nl",
            "ar",
            "sv",
            "it",
            "id",
            "hi",
            "fi",
            "vi",
            "he",
            "uk",
            "el",
            "ms",
            "cs",
            "ro",
            "da",
            "hu",
            "ta",
            "no",
            "th",
            "ur",
            "hr",
            "bg",
            "lt",
            "la",
            "mi",
            "ml",
            "cy",
            "sk",
            "te",
            "fa",
            "lv",
            "bn",
            "sr",
            "az",
            "sl",
            "kn",
            "et",
            "mk",
            "br",
            "eu",
            "is",
            "hy",
            "ne",
            "mn",
            "bs",
            "kk",
            "sq",
            "sw",
            "gl",
            "mr",
            "pa",
            "si",
            "km",
            "sn",
            "yo",
            "so",
            "af",
            "oc",
            "ka",
            "be",
            "tg",
            "sd",
            "gu",
            "am",
            "yi",
            "lo",
            "uz",
            "fo",
            "ht",
            "ps",
            "tk",
            "nn",
            "mt",
            "sa",
            "lb",
            "my",
            "bo",
            "tl",
            "mg",
            "as",
            "tt",
            "haw",
            "ln",
            "ha",
            "ba",
            "jw",
            "su",
            "yue"
    };

    private OrtSession session;
    private OrtSession initSession;
    private OrtSession encoderSession;
    private OrtSession cacheInitSession;
    private OrtSession cacheInitBatchSession;
    private OrtSession decoderInitSession;
    private OrtSession decoderSession;
    private OrtSession phoEncoderSession;
    private OrtSession phoDecoderSession;
    private OrtSession detokenizerSession;
    private OrtEnvironment onnxEnv;
    private boolean phoWhisperAvailable = false;
    private boolean legacyWhisperAvailable = false;


    public Recognizer(Global global, final boolean returnResultOnlyAtTheEnd, final NeuralNetworkApi.InitListener initListener) {
        this.global = global;
        //onnxEnv = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE);
        onnxEnv = OrtEnvironment.getEnvironment();

        String filesPath = global.getFilesDir().getPath();
        String modelInitPath = filesPath + "/Whisper_initializer.onnx";
        String detokenizerPath = filesPath + "/Whisper_detokenizer.onnx";
        String legacyEncoderPath = filesPath + "/" + LEGACY_WHISPER_ENCODER_MODEL;
        String legacyDecoderPath = filesPath + "/" + LEGACY_WHISPER_DECODER_MODEL;
        String legacyCacheInitPath = filesPath + "/" + LEGACY_WHISPER_CACHE_INIT_MODEL;
        String legacyCacheInitBatchPath = filesPath + "/" + LEGACY_WHISPER_CACHE_INIT_BATCH_MODEL;
        String phoWhisperEncoderPath = filesPath + "/" + PHOWHISPER_ENCODER_MODEL;
        String phoWhisperDecoderInitPath = filesPath + "/" + PHOWHISPER_DECODER_INIT_MODEL;
        String phoWhisperDecoderPath = filesPath + "/" + PHOWHISPER_DECODER_MODEL;
        phoWhisperAvailable = isPhoWhisperModelSetAvailable(global.getFilesDir());
        legacyWhisperAvailable = isLegacyWhisperModelSetAvailable(global.getFilesDir());

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    OrtSession.SessionOptions initSessionOptions = new OrtSession.SessionOptions();
                    initSessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
                    initSessionOptions.setCPUArenaAllocator(false);
                    initSessionOptions.setMemoryPatternOptimization(false);
                    initSessionOptions.setOptimizationLevel(WHISPER_GRAPH_OPT_LEVEL);
                    initSession = onnxEnv.createSession(modelInitPath, initSessionOptions);

                    OrtSession.SessionOptions encoderSessionOptions = new OrtSession.SessionOptions();
                    encoderSessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
                    if(global.getTotalRamSize() <= 7000){
                        encoderSessionOptions.setCPUArenaAllocator(false);
                        encoderSessionOptions.setMemoryPatternOptimization(false);
                    }else {
                        encoderSessionOptions.setCPUArenaAllocator(true);
                        encoderSessionOptions.setMemoryPatternOptimization(true);
                    }
                    encoderSessionOptions.setOptimizationLevel(WHISPER_GRAPH_OPT_LEVEL);
                    if (legacyWhisperAvailable) {
                        encoderSessionOptions.setSymbolicDimensionValue("batch_size", 1);
                        encoderSession = onnxEnv.createSession(legacyEncoderPath, encoderSessionOptions);
                    }
                    if (phoWhisperAvailable) {
                        OrtSession.SessionOptions phoEncoderSessionOptions = new OrtSession.SessionOptions();
                        phoEncoderSessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
                        if(global.getTotalRamSize() <= 7000){
                            phoEncoderSessionOptions.setCPUArenaAllocator(false);
                            phoEncoderSessionOptions.setMemoryPatternOptimization(false);
                        }else {
                            phoEncoderSessionOptions.setCPUArenaAllocator(true);
                            phoEncoderSessionOptions.setMemoryPatternOptimization(true);
                        }
                        phoEncoderSessionOptions.setSymbolicDimensionValue("batch_size", 1);
                        phoEncoderSessionOptions.setOptimizationLevel(WHISPER_GRAPH_OPT_LEVEL);
                        phoEncoderSession = onnxEnv.createSession(phoWhisperEncoderPath, phoEncoderSessionOptions);
                    }

                    OrtSession.SessionOptions cacheSessionOptions = new OrtSession.SessionOptions();
                    cacheSessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
                    cacheSessionOptions.setCPUArenaAllocator(false);
                    cacheSessionOptions.setMemoryPatternOptimization(false);
                    cacheSessionOptions.setOptimizationLevel(WHISPER_GRAPH_OPT_LEVEL);
                    if (legacyWhisperAvailable) {
                        cacheInitSession = onnxEnv.createSession(legacyCacheInitPath, cacheSessionOptions);
                        cacheInitBatchSession = onnxEnv.createSession(legacyCacheInitBatchPath, cacheSessionOptions);
                    }
                    if (phoWhisperAvailable) {
                        decoderInitSession = onnxEnv.createSession(phoWhisperDecoderInitPath, cacheSessionOptions);
                    }

                    OrtSession.SessionOptions decoderSessionOptions = new OrtSession.SessionOptions();
                    decoderSessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
                    decoderSessionOptions.setCPUArenaAllocator(false);
                    decoderSessionOptions.setMemoryPatternOptimization(false);
                    decoderSessionOptions.setOptimizationLevel(WHISPER_GRAPH_OPT_LEVEL);
                    if (legacyWhisperAvailable) {
                        decoderSession = onnxEnv.createSession(legacyDecoderPath, decoderSessionOptions);
                    }
                    if (phoWhisperAvailable) {
                        OrtSession.SessionOptions phoDecoderSessionOptions = new OrtSession.SessionOptions();
                        phoDecoderSessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
                        phoDecoderSessionOptions.setCPUArenaAllocator(false);
                        phoDecoderSessionOptions.setMemoryPatternOptimization(false);
                        phoDecoderSessionOptions.setOptimizationLevel(WHISPER_GRAPH_OPT_LEVEL);
                        phoDecoderSession = onnxEnv.createSession(phoWhisperDecoderPath, phoDecoderSessionOptions);
                    }

                    OrtSession.SessionOptions detokenizerSessionOptions = new OrtSession.SessionOptions();
                    detokenizerSessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
                    detokenizerSessionOptions.setCPUArenaAllocator(false);
                    detokenizerSessionOptions.setMemoryPatternOptimization(false);
                    //detokenizerSessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT);
                    detokenizerSession = onnxEnv.createSession(detokenizerPath, detokenizerSessionOptions);

                    Log.i("recognizer", "using language-specific ASR routing. PhoWhisper available: " + phoWhisperAvailable + ", legacy Whisper available: " + legacyWhisperAvailable);
                    initListener.onInitializationFinished();
                } catch (OrtException e) {
                    e.printStackTrace();
                    initListener.onError(new int[]{ErrorCodes.ERROR_LOADING_MODEL},0);
                }
            }
        }).start();
    }

    /**
     * Recognizes the speech audio. This method should be called every time a chunk of byte buffer
     * is returned from onVoice.
     *
     * @param data The audio data.
     */
    public void recognize(final float[] data, int beamSize, final String languageCode) {
        new Thread("recognizer"){
            @Override
            public void run() {
                super.run();
                synchronized (lock) {
                    Log.e("recognizer","recognizingCalled");
                    if (data != null) {
                        int effectiveBeamSize = getEffectiveBeamSize(beamSize, false);
                        dataToRecognize.addLast(new DataContainer(data, effectiveBeamSize, languageCode));
                        if (dataToRecognize.size() >= 1 && !recognizing) {
                            recognize();
                        }
                    }
                }
            }
        }.start();
    }

    public void recognize(final float[] data, int beamSize, final String languageCode1, final String languageCode2) {
        new Thread("recognizer"){
            @Override
            public void run() {
                super.run();
                synchronized (lock) {
                    Log.e("recognizer","recognizingCalled");
                    if (data != null) {
                        int effectiveBeamSize = getEffectiveBeamSize(beamSize, true);
                        int refinementBeamSize = getRefinementBeamSize(beamSize, true);
                        if (refinementBeamSize > 1) {
                            Log.i("recognizer", "automatic multi-language recognition uses greedy language pass + winner beam refinement: " + refinementBeamSize);
                        }
                        dataToRecognize.addLast(new DataContainer(data, effectiveBeamSize, refinementBeamSize, languageCode1, languageCode2));
                        if (dataToRecognize.size() >= 1 && !recognizing) {
                            recognize();
                        }
                    }
                }
            }
        }.start();
    }

    private void recognize() {
        recognizing = true;
        DataContainer data = dataToRecognize.pollFirst();
        if (isModelReady()) {
            if (data != null) {
                //we convert data in un audioTensor and start the transcription
                try {
                    saveDebugAudio(data.data, "debug_last_speech_raw.wav");
                    float[] audioData = preprocessAudioForRecognition(data.data, Recorder.SAMPLE_RATE_CANDIDATES[0]);
                    saveDebugAudio(audioData, "debug_last_speech.wav");
                    FloatBuffer floatAudioDataBuffer = FloatBuffer.wrap(audioData);
                    OnnxTensor audioTensor = OnnxTensor.createTensor(onnxEnv, floatAudioDataBuffer, TensorUtils.tensorShape(1L, (long) audioData.length));

                    // if we generate more than this number of tokens it means that we have an infinite loop due to the fact that the sound cannot be transcribed with the language selected
                    int maxTokens = calculateMaxTokens(audioData.length);
                    boolean execution1HitMaxLength = false;
                    boolean execution2HitMaxLength = false;

                    //execution of pre ops
                    long time = System.currentTimeMillis();
                    long startTimeInMs = SystemClock.elapsedRealtime();
                    Map initInputs = (Map) (new LinkedHashMap());
                    initInputs.put("audio_pcm", audioTensor);
                    OrtSession.Result outputsInit = initSession.run(initInputs);
                    OnnxTensor outputInit = (OnnxTensor) outputsInit.get(0);
                    android.util.Log.i("performance", "pre ops done in: " + (System.currentTimeMillis()-time) + "ms");

                    DetectedLanguage detectedLanguage = null;
                    if (data.languageCode2 != null) {
                        detectedLanguage = detectSpokenLanguage(outputInit, data.languageCode, data.languageCode2);
                    }

                    if (detectedLanguage != null) {
                        recognizeDetectedLanguageOnly(data, outputInit, detectedLanguage, maxTokens);
                        outputInit.close();
                        android.util.Log.i("performance", "SPEECH RECOGNITION DONE IN: " + (SystemClock.elapsedRealtime() - startTimeInMs) + "ms");
                    } else if (usesMixedWhisperEngines(data)) {
                        recognizeWithMixedWhisperEngines(data, outputInit, maxTokens);
                        outputInit.close();
                        android.util.Log.i("performance", "SPEECH RECOGNITION DONE IN: " + (SystemClock.elapsedRealtime() - startTimeInMs) + "ms");
                    } else {
                    //execution of the encoder
                    boolean usePhoWhisperForRequest = shouldUsePhoWhisperForLanguage(data.languageCode);
                    android.util.Log.i("recognizer", "selected ASR engine for " + data.languageCode + (data.languageCode2 == null ? "" : "/" + data.languageCode2) + ": " + (usePhoWhisperForRequest ? "PhoWhisper" : "Whisper"));
                    Map inputs = (Map) (new LinkedHashMap());
                    inputs.put("input_features", outputInit);
                    OrtSession.Result outputs = (usePhoWhisperForRequest ? phoEncoderSession : encoderSession).run(inputs);
                    OnnxTensor outputEncoder = (OnnxTensor) outputs.get(0);
                    android.util.Log.i("performance", "Encoder done in: " + (SystemClock.elapsedRealtime() - startTimeInMs) + "ms");

                    if (usePhoWhisperForRequest) {
                        recognizeWithPhoWhisper(data, outputEncoder, maxTokens);
                        outputs.close();
                        outputInit.close();
                        android.util.Log.i("performance", "SPEECH RECOGNITION DONE IN: " + (SystemClock.elapsedRealtime() - startTimeInMs) + "ms");
                    } else {
                    //execution of decoder
                    final int eos = 50257;
                    ArrayList<Integer> completeOutput = new ArrayList<Integer>();
                    ArrayList<Integer> completeOutput2 = new ArrayList<Integer>();
                    double outputProbability1 = 0;
                    double outputProbability2 = 0;
                    boolean finished1 = false;
                    boolean finished2 = false;
                    boolean execution1NoSpeech = false;
                    boolean execution2NoSpeech = false;
                    double noSpeechProbability1 = 0;
                    double noSpeechProbability2 = 0;
                    long initialTime;

                    OnnxTensor inputIDsTensor = null;
                    OnnxTensor decoderOutput = null;
                    Map<String, OnnxTensor> decoderInput = new HashMap<String, OnnxTensor>();
                    float[][][] value = null;
                    float[] outputValues = null;
                    float[] outputValues2 = null;
                    int batchSize = 1;

                    if(data.languageCode2 != null){
                        batchSize = 2;
                    }

                    //We prepare the cache initializer input and execute it
                    Map<String, OnnxTensor> initInput = new HashMap<String, OnnxTensor>();
                    OrtSession.Result initResult = null;
                    if(batchSize == 1 || data.beamSize > 1) {
                        initInput.put("encoder_hidden_states", outputEncoder);
                        time = System.currentTimeMillis();
                        initResult = cacheInitSession.run(initInput);
                        android.util.Log.i("performance", "Cache initialization done in: " + (System.currentTimeMillis() - time) + "ms");
                    }else{
                        long timeInner = System.currentTimeMillis();
                        float[][] outputEncoderValue = ((float[][][]) outputEncoder.getValue())[0];
                        android.util.Log.i("performance", "Encoder batch extract done in: " + (System.currentTimeMillis()-timeInner) + "ms");
                        timeInner = System.currentTimeMillis();
                        float[][][] outputEncoderFlatBatched = TensorUtils.batchTensor(outputEncoderValue, 2);
                        android.util.Log.i("performance", "Encoder batch batching done in: " + (System.currentTimeMillis()-timeInner) + "ms");
                        timeInner = System.currentTimeMillis();
                        OnnxTensor outputEncoderBatched = TensorUtils.createFloatTensor(onnxEnv, outputEncoderFlatBatched, new long[]{2, outputEncoderValue.length, outputEncoderValue[0].length}, new long[]{0});
                        android.util.Log.i("performance", "Encoder batch creation done in: " + (System.currentTimeMillis()-timeInner) + "ms");
                        initInput.put("encoder_hidden_states", outputEncoderBatched);
                        time = System.currentTimeMillis();
                        initResult = cacheInitBatchSession.run(initInput);
                        android.util.Log.i("performance", "Cache initialization done in: " + (System.currentTimeMillis() - time) + "ms");
                    }

                    //We start the iterative execution of the decoder
                    OrtSession.Result result = null;
                    OrtSession.Result oldResult = null;
                    int max = -1;
                    int max2 = eos;
                    boolean isFirstIteration = true;  //It is used to avoid closing initialResult
                    int j = 1;
                    int scoreTokenCount1 = 0;
                    int scoreTokenCount2 = 0;
                    //first we run the decoder with the fixed prompt values (START_TOKEN_ID, languageID, TRANSCRIBE_TOKEN_ID, NO_TIMESTAMPS_TOKEN_ID)
                    int languageID = getLanguageID(data.languageCode);
                    int languageID2 = -1;
                    if(batchSize == 2){
                        languageID2 = getLanguageID(data.languageCode2);
                    }
                    int[] decoderInitialInputIDs = {START_TOKEN_ID, languageID, TRANSCRIBE_TOKEN_ID, NO_TIMESTAMPS_TOKEN_ID};
                    int[] decoderInitialInputIDs2 = {START_TOKEN_ID, languageID2, TRANSCRIBE_TOKEN_ID, NO_TIMESTAMPS_TOKEN_ID};

                    if(data.beamSize > 1) {
                        DecodeResult beamResult = executeCacheDecoderBeam(initResult, decoderInitialInputIDs, data.beamSize, maxTokens);
                        completeOutput = beamResult.output;
                        outputProbability1 = beamResult.averageLogProbability;
                        noSpeechProbability1 = beamResult.noSpeechProbability;
                        execution1HitMaxLength = beamResult.hitMaxLength;
                        if(batchSize == 2) {
                            DecodeResult beamResult2 = executeCacheDecoderBeam(initResult, decoderInitialInputIDs2, data.beamSize, maxTokens);
                            completeOutput2 = beamResult2.output;
                            outputProbability2 = beamResult2.averageLogProbability;
                            noSpeechProbability2 = beamResult2.noSpeechProbability;
                            execution2HitMaxLength = beamResult2.hitMaxLength;
                        }
                    }else {
                    while (!(max == eos && max2 == eos)) {
                        initialTime = System.currentTimeMillis();
                        time = System.currentTimeMillis();
                        if (j <= 4) {
                            if(batchSize == 1) {
                                inputIDsTensor = TensorUtils.convertIntArrayToTensor(onnxEnv, new int[]{decoderInitialInputIDs[j-1]});
                            }else{
                                inputIDsTensor = TensorUtils.convertIntArrayToTensor(onnxEnv, new int[]{decoderInitialInputIDs[j-1], decoderInitialInputIDs2[j-1]}, new long[]{2,1});
                            }
                        }
                        //We prepare the decoder input
                        decoderInput = new HashMap<String, OnnxTensor>();
                        decoderInput.put("input_ids", inputIDsTensor);
                        if (isFirstIteration) {
                            long[] shape = {batchSize, 12, 0, 64};
                            OnnxTensor decoderPastTensor = TensorUtils.createFloatTensorWithSingleValue(onnxEnv, 0, shape);
                            for (int i = 0; i < 12; i++) {
                                decoderInput.put("past_key_values." + i + ".decoder.key", decoderPastTensor);
                                decoderInput.put("past_key_values." + i + ".decoder.value", decoderPastTensor);
                                decoderInput.put("past_key_values." + i + ".encoder.key", (OnnxTensor) initResult.get("present." + i + ".encoder.key").get());
                                decoderInput.put("past_key_values." + i + ".encoder.value", (OnnxTensor) initResult.get("present." + i + ".encoder.value").get());
                            }
                            isFirstIteration = false;
                        } else {
                            for (int i = 0; i < 12; i++) {
                                decoderInput.put("past_key_values." + i + ".decoder.key", (OnnxTensor) result.get("present." + i + ".decoder.key").get());
                                decoderInput.put("past_key_values." + i + ".decoder.value", (OnnxTensor) result.get("present." + i + ".decoder.value").get());
                                decoderInput.put("past_key_values." + i + ".encoder.key", (OnnxTensor) initResult.get("present." + i + ".encoder.key").get());
                                decoderInput.put("past_key_values." + i + ".encoder.value", (OnnxTensor) initResult.get("present." + i + ".encoder.value").get());
                            }
                        }
                        oldResult = result;
                        android.util.Log.i("performance", "pre-execution of" + j + "th word done in: " + (System.currentTimeMillis() - time) + "ms");
                        time = System.currentTimeMillis();
                        //execution of decoder (with cache)
                        result = decoderSession.run(decoderInput);
                        android.util.Log.i("performance", "execution of" + j + "th word done in: " + (System.currentTimeMillis() - time) + "ms");
                        time = System.currentTimeMillis();

                        if (oldResult != null) {
                            oldResult.close(); //serve a rilasciare la memoria occupata dal risultato (altrimenti di accumula e aumenta molto)
                            android.util.Log.i("performance", "release RAM of" + j + "th word done in: " + (System.currentTimeMillis() - time) + "ms");
                        }
                        //we extract the logits and the highest value
                        decoderOutput = (OnnxTensor) result.get("logits").get();
                        value = (float[][][]) decoderOutput.getValue();
                        outputValues = value[0][0];
                        if(j == 1) {
                            noSpeechProbability1 = tokenProbability(outputValues[NO_SPEECH_TOKEN_ID], outputValues);
                        }
                        boolean isPromptStep = j < decoderInitialInputIDs.length;
                        if(!finished1) {
                            if(isPromptStep) {
                                max = decoderInitialInputIDs[j];
                            }else {
                                applyTextLogitFilters(outputValues, j == decoderInitialInputIDs.length);
                                max = Utils.getIndexOfLargest(outputValues);
                                completeOutput.add(max);
                            }
                        }
                        if(batchSize == 2){
                            outputValues2 = value[1][0];
                            if(j == 1) {
                                noSpeechProbability2 = tokenProbability(outputValues2[NO_SPEECH_TOKEN_ID], outputValues2);
                            }
                            if(!finished2) {
                                if(isPromptStep) {
                                    max2 = decoderInitialInputIDs2[j];
                                }else {
                                    applyTextLogitFilters(outputValues2, j == decoderInitialInputIDs.length);
                                    max2 = Utils.getIndexOfLargest(outputValues2);
                                    completeOutput2.add(max2);
                                }
                            }
                        }
                        //We prepare the inputs for the next iteration
                        if(batchSize == 1) {
                            inputIDsTensor = TensorUtils.convertIntArrayToTensor(onnxEnv, new int[]{max});
                        }else{
                            inputIDsTensor = TensorUtils.convertIntArrayToTensor(onnxEnv, new int[]{max, max2}, new long[]{2,1});
                        }

                        if(!isPromptStep) {
                            //we calculate and update the probabilities of the output sentences
                            if(!finished1) {
                                outputProbability1 = outputProbability1 + tokenLogProbability(outputValues[max], outputValues);
                                scoreTokenCount1++;
                            }
                            if(batchSize == 2 && !finished2) {
                                outputProbability2 = outputProbability2 + tokenLogProbability(outputValues2[max2], outputValues2);
                                scoreTokenCount2++;
                            }
                        }

                        if(!isPromptStep && j - (decoderInitialInputIDs.length - 1) >= maxTokens) {
                            if (!finished1) {
                                execution1HitMaxLength = true;
                                max = eos;
                            }
                            if (!finished2) {
                                execution2HitMaxLength = true;
                                max2 = eos;
                            }
                        }

                        if(max == eos){
                            finished1 = true;
                        }
                        if(max2 == eos){
                            finished2 = true;
                        }
                        android.util.Log.i("performance", "post-execution of" + j + "th word done in: " + (System.currentTimeMillis() - time) + "ms");
                        android.util.Log.i("performance", "Generation of" + j + "th word done in: " + (System.currentTimeMillis() - initialTime) + "ms");

                        j++;
                    }
                    }

                    //we normalize the scores based on the number of tokens of the respective sequence
                    if(data.beamSize > 1) {
                        //beam search already returns a normalized score.
                    }else {
                        outputProbability1 = normalizeLogProbability(outputProbability1, scoreTokenCount1);
                    }
                    if(batchSize == 2 && data.beamSize <= 1) {
                        outputProbability2 = normalizeLogProbability(outputProbability2, scoreTokenCount2);
                    }

                    if(batchSize == 2 && data.beamSize <= 1 && data.refinementBeamSize > 1) {
                        boolean preliminaryExecution1NoSpeech = isNoSpeech(noSpeechProbability1, outputProbability1);
                        boolean preliminaryExecution2NoSpeech = isNoSpeech(noSpeechProbability2, outputProbability2);
                        int refinementLanguageIndex = selectAutoRefinementLanguage(execution1HitMaxLength, preliminaryExecution1NoSpeech, outputProbability1, execution2HitMaxLength, preliminaryExecution2NoSpeech, outputProbability2);
                        if(refinementLanguageIndex == 1) {
                            DecodeResult refinedResult = executeCacheDecoderBeam(outputEncoder, decoderInitialInputIDs, data.refinementBeamSize, maxTokens);
                            completeOutput = refinedResult.output;
                            outputProbability1 = refinedResult.averageLogProbability;
                            noSpeechProbability1 = refinedResult.noSpeechProbability;
                            execution1HitMaxLength = refinedResult.hitMaxLength;
                        }else if(refinementLanguageIndex == 2) {
                            DecodeResult refinedResult = executeCacheDecoderBeam(outputEncoder, decoderInitialInputIDs2, data.refinementBeamSize, maxTokens);
                            completeOutput2 = refinedResult.output;
                            outputProbability2 = refinedResult.averageLogProbability;
                            noSpeechProbability2 = refinedResult.noSpeechProbability;
                            execution2HitMaxLength = refinedResult.hitMaxLength;
                        }
                    }
                    execution1NoSpeech = isNoSpeech(noSpeechProbability1, outputProbability1);
                    if(batchSize == 2) {
                        execution2NoSpeech = isNoSpeech(noSpeechProbability2, outputProbability2);
                    }
                    android.util.Log.i("result", "no_speech 1: " + noSpeechProbability1);
                    if(batchSize == 2) {
                        android.util.Log.i("result", "no_speech 2: " + noSpeechProbability2);
                    }

                    //execution of the detokenizer
                    Map detokenizerInputs = (Map) (new LinkedHashMap());
                    if(batchSize == 1) {
                        String finalText = UNDEFINED_TEXT;
                        if(!execution1HitMaxLength && !execution1NoSpeech) {
                            int[] sequences = completeOutput.stream().mapToInt(i -> i).toArray();
                            detokenizerInputs.put("sequences", TensorUtils.createInt32Tensor(onnxEnv, sequences, new long[]{1, 1, sequences.length}));
                            OrtSession.Result detokenizerOutputs = this.detokenizerSession.run(detokenizerInputs);
                            Object finalTextResult = detokenizerOutputs.get(0).getValue();
                            finalText = ((String[][]) finalTextResult)[0][0];
                            detokenizerOutputs.close();
                        }
                        finalText = correctBasicText(finalText);
                        android.util.Log.i("result", "result: " + finalText);
                        android.util.Log.i("score", "score: " + outputProbability1);

                        notifyResult(finalText, data.languageCode, outputProbability1, true);

                    }else{
                        String firstText = UNDEFINED_TEXT;
                        if(!execution1HitMaxLength && !execution1NoSpeech) {
                            int[] sequence1 = completeOutput.stream().mapToInt(i -> i).toArray();
                            detokenizerInputs.put("sequences", OnnxTensor.createTensor(onnxEnv, IntBuffer.wrap(sequence1), TensorUtils.tensorShape(1, 1, sequence1.length)));
                            OrtSession.Result detokenizerOutputs = this.detokenizerSession.run(detokenizerInputs);
                            Object firstTextResult = detokenizerOutputs.get(0).getValue();
                            firstText = ((String[][]) firstTextResult)[0][0];
                            detokenizerOutputs.close();
                        }

                        String secondText = UNDEFINED_TEXT;
                        if(!execution2HitMaxLength && !execution2NoSpeech) {
                            int[] sequence2 = completeOutput2.stream().mapToInt(i -> i).toArray();
                            detokenizerInputs = (Map) (new LinkedHashMap());
                            detokenizerInputs.put("sequences", OnnxTensor.createTensor(onnxEnv, IntBuffer.wrap(sequence2), TensorUtils.tensorShape(1, 1, sequence2.length)));
                            OrtSession.Result detokenizerOutputs2 = this.detokenizerSession.run(detokenizerInputs);
                            Object secondTextResult = detokenizerOutputs2.get(0).getValue();
                            secondText = ((String[][]) secondTextResult)[0][0];
                            detokenizerOutputs2.close();
                        }

                        firstText = correctBasicText(firstText);
                        secondText = correctBasicText(secondText);
                        android.util.Log.i("result", "result 1: " + firstText);
                        android.util.Log.i("result", "result 2: " + secondText);
                        android.util.Log.i("result", "score 1: " + outputProbability1);
                        android.util.Log.i("result", "score 2: " + outputProbability2);

                        notifyMultiResult(firstText, data.languageCode, outputProbability1, secondText, data.languageCode2, outputProbability2);
                    }
                    //closing all results
                    outputs.close();
                    outputInit.close();
                    initResult.close();

                    android.util.Log.i("performance", "SPEECH RECOGNITION DONE IN: " + (SystemClock.elapsedRealtime() - startTimeInMs) + "ms");
                    }
                    }

                } catch (OrtException e) {
                    e.printStackTrace();
                    notifyError(new int[]{ErrorCodes.ERROR_EXECUTING_MODEL}, 0);
                }
            }
        }
        if (!dataToRecognize.isEmpty()){
            recognize();
        }else {
            recognizing = false;
        }
    }

    private boolean isModelReady() {
        return initSession != null
                && detokenizerSession != null
                && isLegacyWhisperReady()
                && isPhoWhisperReady();
    }

    private boolean isLegacyWhisperReady() {
        return encoderSession != null
                && cacheInitSession != null
                && cacheInitBatchSession != null
                && decoderSession != null;
    }

    private boolean isPhoWhisperReady() {
        return phoEncoderSession != null
                && decoderInitSession != null
                && phoDecoderSession != null;
    }

    private static String correctBasicText(String text){
        String correctedText = text;
        if (correctedText == null) {
            return "";
        }

        //sometimes, even if timestamps are deactivated, Whisper insert those anyway (es. <|0.00|>), so we remove eventual timestamps
        String regex = "<\\|[^>]*\\|> ";    //with this regex we remove all substrings of the form "<|something|> "
        correctedText = correctedText.replaceAll(regex, "");

        //we remove eventual white space from both ends of the text
        correctedText = correctedText.trim();

        if(correctedText.length() >= 2) {
            //if the correctedText start with a lower case letter we make it upper case
            char firstChar = correctedText.charAt(0);
            if (Character.isLowerCase(firstChar)) {
                StringBuilder sb = new StringBuilder(correctedText);
                sb.setCharAt(0, Character.toUpperCase(firstChar));
                correctedText = sb.toString();
            }
            //if the correctedText contains a "..." we remove it
            correctedText = correctedText.replace("...", "");
        }
        return correctedText;
    }

    // this method returns only the languages of Whisper-small that have a minimum quality (wer <= 37%)
    // LANGUAGES instead contains all languages supported by Whisper and it is needed for generating the language ID
    public static ArrayList<CustomLocale> getSupportedLanguages(Context context) {
        ArrayList<CustomLocale> languages = new ArrayList<>();
        SharedPreferences sharedPreferences = context.getSharedPreferences("default", Context.MODE_PRIVATE);
        boolean qualityLow = sharedPreferences.getBoolean("languagesNNQualityLow", false);
        if(!qualityLow) {
            DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory.newInstance();
            try {
                DocumentBuilder documentBuilder = documentBuilderFactory.newDocumentBuilder();
                Document document = documentBuilder.parse(context.getResources().openRawResource(R.raw.whisper_supported_languages));
                NodeList list = document.getElementsByTagName("code");
                for (int i = 0; i < list.getLength(); i++) {
                    languages.add(CustomLocale.getInstance(list.item(i).getTextContent()));
                }
            } catch (IOException | SAXException | ParserConfigurationException e) {
                e.printStackTrace();
            }
        }else{
            for (String language : LANGUAGES) {
                languages.add(CustomLocale.getInstance(language));
            }
        }
        return languages;
    }

    public void destroy() {
        //eventually if in the future I decide to load Whisper only for WalkieTalkie and Conversation then all the resources will be released here
    }

    private DetectedLanguage detectSpokenLanguage(OnnxTensor outputInit, String firstLanguageCode, String secondLanguageCode) throws OrtException {
        int firstLanguageID = getLanguageID(firstLanguageCode);
        int secondLanguageID = getLanguageID(secondLanguageCode);
        if (firstLanguageID < 0 || secondLanguageID < 0 || firstLanguageID == secondLanguageID || !isLegacyWhisperReady()) {
            return null;
        }

        OrtSession.Result outputs = null;
        OrtSession.Result initResult = null;
        OrtSession.Result result = null;
        try {
            Map inputs = (Map) (new LinkedHashMap());
            inputs.put("input_features", outputInit);
            outputs = encoderSession.run(inputs);
            OnnxTensor outputEncoder = (OnnxTensor) outputs.get(0);

            Map<String, OnnxTensor> initInput = new HashMap<>();
            initInput.put("encoder_hidden_states", outputEncoder);
            initResult = cacheInitSession.run(initInput);

            OnnxTensor inputIDsTensor = TensorUtils.convertIntArrayToTensor(onnxEnv, new int[]{START_TOKEN_ID});
            Map<String, OnnxTensor> decoderInput = createDecoderInput(inputIDsTensor, null, initResult, 1, true);
            result = decoderSession.run(decoderInput);

            OnnxTensor decoderOutput = (OnnxTensor) result.get("logits").get();
            float[] outputValues = ((float[][][]) decoderOutput.getValue())[0][0];
            double firstProbability = normalizedCandidateProbability(outputValues, firstLanguageID, secondLanguageID, firstLanguageID);
            double secondProbability = normalizedCandidateProbability(outputValues, firstLanguageID, secondLanguageID, secondLanguageID);
            int bestLanguageID = getBestLanguageTokenID(outputValues);
            double bestLanguageProbability = languageTokenProbability(outputValues, bestLanguageID);
            int detectedIndex = selectAudioDetectedLanguageIndex(outputValues, firstLanguageID, secondLanguageID);
            String detectedLanguageCode = detectedIndex == 1 ? firstLanguageCode : detectedIndex == 2 ? secondLanguageCode : null;
            android.util.Log.i("recognizer", "audio language detection top=" + getLanguageCodeFromLanguageTokenID(bestLanguageID)
                    + "=" + bestLanguageProbability + ", " + firstLanguageCode + "=" + firstProbability
                    + ", " + secondLanguageCode + "=" + secondProbability + ", selected=" + detectedLanguageCode);
            if (detectedIndex == 0) {
                return null;
            }
            return new DetectedLanguage(detectedIndex, detectedLanguageCode);
        } finally {
            if (result != null) {
                result.close();
            }
            if (initResult != null) {
                initResult.close();
            }
            if (outputs != null) {
                outputs.close();
            }
        }
    }

    private void recognizeDetectedLanguageOnly(DataContainer data, OnnxTensor outputInit, DetectedLanguage detectedLanguage, int maxTokens) throws OrtException {
        RecognizedResult recognizedResult = recognizeSingleLanguageWithSelectedWhisper(outputInit, detectedLanguage.languageCode, data.beamSize, maxTokens);
        if (detectedLanguage.index == 1) {
            android.util.Log.i("result", "audio-selected result 1: " + recognizedResult.text);
            notifyMultiResult(recognizedResult.text, data.languageCode, recognizedResult.averageLogProbability,
                    UNDEFINED_TEXT, data.languageCode2, Double.NEGATIVE_INFINITY);
        } else {
            android.util.Log.i("result", "audio-selected result 2: " + recognizedResult.text);
            notifyMultiResult(UNDEFINED_TEXT, data.languageCode, Double.NEGATIVE_INFINITY,
                    recognizedResult.text, data.languageCode2, recognizedResult.averageLogProbability);
        }
    }

    private boolean usesMixedWhisperEngines(DataContainer data) {
        return data.languageCode2 != null
                && shouldUsePhoWhisperForLanguage(data.languageCode) != shouldUsePhoWhisperForLanguage(data.languageCode2);
    }

    private void recognizeWithMixedWhisperEngines(DataContainer data, OnnxTensor outputInit, int maxTokens) throws OrtException {
        RecognizedResult firstResult = recognizeSingleLanguageWithSelectedWhisper(outputInit, data.languageCode, data.beamSize, maxTokens);
        RecognizedResult secondResult = recognizeSingleLanguageWithSelectedWhisper(outputInit, data.languageCode2, data.beamSize, maxTokens);
        android.util.Log.i("result", "result 1: " + firstResult.text);
        android.util.Log.i("result", "result 2: " + secondResult.text);
        android.util.Log.i("result", "score 1: " + firstResult.averageLogProbability);
        android.util.Log.i("result", "score 2: " + secondResult.averageLogProbability);
        notifyMultiResult(firstResult.text, data.languageCode, firstResult.averageLogProbability, secondResult.text, data.languageCode2, secondResult.averageLogProbability);
    }

    private RecognizedResult recognizeSingleLanguageWithSelectedWhisper(OnnxTensor outputInit, String languageCode, int beamSize, int maxTokens) throws OrtException {
        boolean usePhoWhisper = shouldUsePhoWhisperForLanguage(languageCode);
        if (usePhoWhisper && !isPhoWhisperReady()) {
            Log.e("recognizer", "PhoWhisper model is not ready for language: " + languageCode);
            return new RecognizedResult(UNDEFINED_TEXT, Double.NEGATIVE_INFINITY);
        }
        if (!usePhoWhisper && !isLegacyWhisperReady()) {
            Log.e("recognizer", "Legacy Whisper model is not ready for language: " + languageCode);
            return new RecognizedResult(UNDEFINED_TEXT, Double.NEGATIVE_INFINITY);
        }

        OrtSession.Result outputs = null;
        try {
            Map inputs = (Map) (new LinkedHashMap());
            inputs.put("input_features", outputInit);
            outputs = (usePhoWhisper ? phoEncoderSession : encoderSession).run(inputs);
            OnnxTensor outputEncoder = (OnnxTensor) outputs.get(0);

            int languageID = getLanguageID(languageCode);
            int[] decoderInitialInputIDs = {START_TOKEN_ID, languageID, TRANSCRIBE_TOKEN_ID, NO_TIMESTAMPS_TOKEN_ID};
            DecodeResult decodeResult;
            if (usePhoWhisper) {
                if (beamSize > 1) {
                    decodeResult = executePhoWhisperDecoderBeam(outputEncoder, decoderInitialInputIDs, beamSize, maxTokens);
                } else {
                    decodeResult = executePhoWhisperGreedy(outputEncoder, new int[][]{decoderInitialInputIDs}, maxTokens)[0];
                }
            } else {
                if (beamSize > 1) {
                    decodeResult = executeCacheDecoderBeam(outputEncoder, decoderInitialInputIDs, beamSize, maxTokens);
                } else {
                    decodeResult = executeLegacyWhisperGreedy(outputEncoder, decoderInitialInputIDs, maxTokens);
                }
            }

            boolean executionNoSpeech = isNoSpeech(decodeResult.noSpeechProbability, decodeResult.averageLogProbability);
            android.util.Log.i("result", "engine " + languageCode + ": " + (usePhoWhisper ? "PhoWhisper" : "Whisper"));
            android.util.Log.i("result", "no_speech " + languageCode + ": " + decodeResult.noSpeechProbability);
            String finalText = UNDEFINED_TEXT;
            if (!decodeResult.hitMaxLength && !executionNoSpeech) {
                finalText = detokenizeOutput(decodeResult.output);
            }
            finalText = correctBasicText(finalText);
            return new RecognizedResult(finalText, decodeResult.averageLogProbability);
        } finally {
            if (outputs != null) {
                outputs.close();
            }
        }
    }

    private void recognizeWithPhoWhisper(DataContainer data, OnnxTensor outputEncoder, int maxTokens) throws OrtException {
        int batchSize = data.languageCode2 != null ? 2 : 1;
        int languageID = getLanguageID(data.languageCode);
        int languageID2 = batchSize == 2 ? getLanguageID(data.languageCode2) : -1;
        int[] decoderInitialInputIDs = {START_TOKEN_ID, languageID, TRANSCRIBE_TOKEN_ID, NO_TIMESTAMPS_TOKEN_ID};
        int[] decoderInitialInputIDs2 = {START_TOKEN_ID, languageID2, TRANSCRIBE_TOKEN_ID, NO_TIMESTAMPS_TOKEN_ID};

        DecodeResult decodeResult1;
        DecodeResult decodeResult2 = null;
        if (data.beamSize > 1) {
            decodeResult1 = executePhoWhisperDecoderBeam(outputEncoder, decoderInitialInputIDs, data.beamSize, maxTokens);
            if (batchSize == 2) {
                decodeResult2 = executePhoWhisperDecoderBeam(outputEncoder, decoderInitialInputIDs2, data.beamSize, maxTokens);
            }
        } else {
            DecodeResult[] greedyResults = executePhoWhisperGreedy(outputEncoder, batchSize == 2 ? new int[][]{decoderInitialInputIDs, decoderInitialInputIDs2} : new int[][]{decoderInitialInputIDs}, maxTokens);
            decodeResult1 = greedyResults[0];
            if (batchSize == 2) {
                decodeResult2 = greedyResults[1];
            }
        }

        if (batchSize == 2 && data.beamSize <= 1 && data.refinementBeamSize > 1) {
            boolean preliminaryExecution1NoSpeech = isNoSpeech(decodeResult1.noSpeechProbability, decodeResult1.averageLogProbability);
            boolean preliminaryExecution2NoSpeech = isNoSpeech(decodeResult2.noSpeechProbability, decodeResult2.averageLogProbability);
            int refinementLanguageIndex = selectAutoRefinementLanguage(decodeResult1.hitMaxLength, preliminaryExecution1NoSpeech, decodeResult1.averageLogProbability,
                    decodeResult2.hitMaxLength, preliminaryExecution2NoSpeech, decodeResult2.averageLogProbability);
            if (refinementLanguageIndex == 1) {
                decodeResult1 = executePhoWhisperDecoderBeam(outputEncoder, decoderInitialInputIDs, data.refinementBeamSize, maxTokens);
            } else if (refinementLanguageIndex == 2) {
                decodeResult2 = executePhoWhisperDecoderBeam(outputEncoder, decoderInitialInputIDs2, data.refinementBeamSize, maxTokens);
            }
        }

        boolean execution1NoSpeech = isNoSpeech(decodeResult1.noSpeechProbability, decodeResult1.averageLogProbability);
        android.util.Log.i("result", "no_speech 1: " + decodeResult1.noSpeechProbability);
        if (batchSize == 1) {
            String finalText = UNDEFINED_TEXT;
            if (!decodeResult1.hitMaxLength && !execution1NoSpeech) {
                finalText = detokenizeOutput(decodeResult1.output);
            }
            finalText = correctBasicText(finalText);
            android.util.Log.i("result", "result: " + finalText);
            android.util.Log.i("score", "score: " + decodeResult1.averageLogProbability);
            notifyResult(finalText, data.languageCode, decodeResult1.averageLogProbability, true);
        } else {
            boolean execution2NoSpeech = isNoSpeech(decodeResult2.noSpeechProbability, decodeResult2.averageLogProbability);
            android.util.Log.i("result", "no_speech 2: " + decodeResult2.noSpeechProbability);
            String firstText = UNDEFINED_TEXT;
            if (!decodeResult1.hitMaxLength && !execution1NoSpeech) {
                firstText = detokenizeOutput(decodeResult1.output);
            }
            String secondText = UNDEFINED_TEXT;
            if (!decodeResult2.hitMaxLength && !execution2NoSpeech) {
                secondText = detokenizeOutput(decodeResult2.output);
            }
            firstText = correctBasicText(firstText);
            secondText = correctBasicText(secondText);
            android.util.Log.i("result", "result 1: " + firstText);
            android.util.Log.i("result", "result 2: " + secondText);
            android.util.Log.i("result", "score 1: " + decodeResult1.averageLogProbability);
            android.util.Log.i("result", "score 2: " + decodeResult2.averageLogProbability);
            notifyMultiResult(firstText, data.languageCode, decodeResult1.averageLogProbability, secondText, data.languageCode2, decodeResult2.averageLogProbability);
        }
    }

    private DecodeResult[] executePhoWhisperGreedy(OnnxTensor outputEncoder, int[][] decoderInitialInputIDs, int maxTokens) throws OrtException {
        int batchSize = decoderInitialInputIDs.length;
        int promptLength = decoderInitialInputIDs[0].length;
        DecodeResult[] decodeResults = new DecodeResult[batchSize];
        ArrayList<Integer>[] completeOutput = new ArrayList[batchSize];
        double[] outputProbability = new double[batchSize];
        double[] noSpeechProbability = new double[batchSize];
        int[] scoreTokenCount = new int[batchSize];
        boolean[] finished = new boolean[batchSize];
        boolean[] hitMaxLength = new boolean[batchSize];
        int[] inputIDs = new int[batchSize];

        OrtSession.Result initResult = null;
        OrtSession.Result result = null;
        OrtSession.Result oldResult = null;
        OnnxTensor batchedOutputEncoder = null;
        try {
            for (int i = 0; i < batchSize; i++) {
                completeOutput[i] = new ArrayList<>();
            }
            OnnxTensor encoderInput = outputEncoder;
            if (batchSize > 1) {
                batchedOutputEncoder = createBatchedOutputEncoderTensor(outputEncoder, batchSize);
                encoderInput = batchedOutputEncoder;
            }

            Map<String, OnnxTensor> initInputs = new HashMap<>();
            initInputs.put("input_ids", createPromptTensor(decoderInitialInputIDs));
            initInputs.put("encoder_hidden_states", encoderInput);
            long time = System.currentTimeMillis();
            initResult = decoderInitSession.run(initInputs);
            android.util.Log.i("performance", "PhoWhisper decoder init done in: " + (System.currentTimeMillis() - time) + "ms");

            float[][][] initLogits = (float[][][]) ((OnnxTensor) initResult.get("logits").get()).getValue();
            for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
                noSpeechProbability[batchIndex] = tokenProbability(initLogits[batchIndex][0][NO_SPEECH_TOKEN_ID], initLogits[batchIndex][0]);
                float[] logits = initLogits[batchIndex][promptLength - 1];
                applyTextLogitFilters(logits, true);
                int token = Utils.getIndexOfLargest(logits);
                completeOutput[batchIndex].add(token);
                outputProbability[batchIndex] += tokenLogProbability(logits[token], logits);
                scoreTokenCount[batchIndex]++;
                finished[batchIndex] = token == EOS_TOKEN_ID;
                inputIDs[batchIndex] = token;
            }

            int generatedTokenCount = 1;
            while (!allFinished(finished) && generatedTokenCount < maxTokens) {
                OnnxTensor inputIDsTensor = TensorUtils.convertIntArrayToTensor(onnxEnv, inputIDs, new long[]{batchSize, 1});
                Map<String, OnnxTensor> decoderInput = createDecoderInput(inputIDsTensor, result == null ? initResult : result, initResult, batchSize, false);
                oldResult = result;
                time = System.currentTimeMillis();
                result = phoDecoderSession.run(decoderInput);
                android.util.Log.i("performance", "PhoWhisper execution of" + generatedTokenCount + "th word done in: " + (System.currentTimeMillis() - time) + "ms");
                if (oldResult != null) {
                    oldResult.close();
                }

                float[][][] outputValues = (float[][][]) ((OnnxTensor) result.get("logits").get()).getValue();
                for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
                    if (finished[batchIndex]) {
                        inputIDs[batchIndex] = EOS_TOKEN_ID;
                        continue;
                    }
                    float[] logits = outputValues[batchIndex][0];
                    applyTextLogitFilters(logits, false);
                    int token = Utils.getIndexOfLargest(logits);
                    completeOutput[batchIndex].add(token);
                    outputProbability[batchIndex] += tokenLogProbability(logits[token], logits);
                    scoreTokenCount[batchIndex]++;
                    finished[batchIndex] = token == EOS_TOKEN_ID;
                    inputIDs[batchIndex] = token;
                }
                generatedTokenCount++;
            }

            if (!allFinished(finished)) {
                for (int i = 0; i < batchSize; i++) {
                    hitMaxLength[i] = !finished[i];
                }
            }
            for (int i = 0; i < batchSize; i++) {
                decodeResults[i] = new DecodeResult(completeOutput[i], normalizeLogProbability(outputProbability[i], scoreTokenCount[i]), noSpeechProbability[i], hitMaxLength[i]);
            }
            return decodeResults;
        } finally {
            if (result != null) {
                result.close();
            }
            if (initResult != null) {
                initResult.close();
            }
            if (batchedOutputEncoder != null) {
                batchedOutputEncoder.close();
            }
        }
    }

    private DecodeResult executePhoWhisperDecoderBeam(OnnxTensor outputEncoder, int[] decoderInitialInputIDs, int requestedBeamSize, int maxTokens) throws OrtException {
        int beamSize = Math.max(2, requestedBeamSize);
        android.util.Log.i("recognizer", "using PhoWhisper beam search: " + beamSize);
        ArrayList<Integer>[] completeBeamOutput = new ArrayList[beamSize];
        for (int i = 0; i < beamSize; i++) {
            completeBeamOutput[i] = new ArrayList<>();
        }
        double[] beamScores = new double[beamSize];
        boolean[] finished = new boolean[beamSize];
        Arrays.fill(beamScores, Double.NEGATIVE_INFINITY);

        OrtSession.Result initResult = null;
        OrtSession.Result result = null;
        OrtSession.Result oldResult = null;
        OrtSession.Result initResultBatched = null;
        CacheContainerNative cacheContainer = null;
        int[] inputIDs = new int[beamSize];
        double noSpeechProbability = 0;
        int generatedTokenCount = 0;
        boolean hitMaxLength = false;
        try {
            Map<String, OnnxTensor> initInputs = new HashMap<>();
            initInputs.put("input_ids", createPromptTensor(new int[][]{decoderInitialInputIDs}));
            initInputs.put("encoder_hidden_states", outputEncoder);
            long time = System.currentTimeMillis();
            initResult = decoderInitSession.run(initInputs);
            android.util.Log.i("performance", "PhoWhisper beam decoder init done in: " + (System.currentTimeMillis() - time) + "ms");

            float[][][] initLogits = (float[][][]) ((OnnxTensor) initResult.get("logits").get()).getValue();
            float[] noSpeechLogits = initLogits[0][0];
            noSpeechProbability = tokenProbability(noSpeechLogits[NO_SPEECH_TOKEN_ID], noSpeechLogits);
            float[] outputValues = initLogits[0][decoderInitialInputIDs.length - 1];
            applyTextLogitFilters(outputValues, true);
            ArrayList<Integer> selectedTokens = new ArrayList<>();
            for (int i = 0; i < beamSize; i++) {
                int token = Utils.getIndexOfLargest(outputValues, selectedTokens);
                selectedTokens.add(token);
                completeBeamOutput[i].add(token);
                beamScores[i] = tokenLogProbability(outputValues[token], outputValues);
                finished[i] = token == EOS_TOKEN_ID;
                inputIDs[i] = token;
            }
            generatedTokenCount = 1;

            initResultBatched = createBatchedEncoderCacheResult(initResult, beamSize);
            result = createBatchedDecoderCacheResult(initResult, beamSize);
            initResult.close();
            initResult = null;

            while (!allFinished(finished) && generatedTokenCount < maxTokens) {
                OnnxTensor inputIDsTensor = TensorUtils.convertIntArrayToTensor(onnxEnv, inputIDs, new long[]{beamSize, 1});
                Map<String, OnnxTensor> decoderInput = createDecoderInput(inputIDsTensor, result, initResultBatched, beamSize, false);
                oldResult = result;
                result = phoDecoderSession.run(decoderInput);
                oldResult.close();

                float[][][] logitsByBeam = (float[][][]) ((OnnxTensor) result.get("logits").get()).getValue();
                ArrayList<BeamCandidate> candidates = new ArrayList<>();
                for (int beamIndex = 0; beamIndex < beamSize; beamIndex++) {
                    if (finished[beamIndex]) {
                        candidates.add(new BeamCandidate(beamIndex, EOS_TOKEN_ID, beamScores[beamIndex], true));
                        continue;
                    }
                    float[] logits = logitsByBeam[beamIndex][0];
                    applyTextLogitFilters(logits, false);
                    selectedTokens = new ArrayList<>();
                    for (int i = 0; i < beamSize; i++) {
                        int token = Utils.getIndexOfLargest(logits, selectedTokens);
                        selectedTokens.add(token);
                        candidates.add(new BeamCandidate(beamIndex, token, beamScores[beamIndex] + tokenLogProbability(logits[token], logits), token == EOS_TOKEN_ID));
                    }
                }

                BeamCandidate[] bestCandidates = selectBestCandidates(candidates, beamSize);
                ArrayList<Integer>[] oldCompleteBeamOutput = completeBeamOutput.clone();
                int[] parentIndexes = new int[beamSize];
                for (int i = 0; i < beamSize; i++) {
                    BeamCandidate candidate = bestCandidates[i];
                    parentIndexes[i] = candidate.parentIndex;
                    completeBeamOutput[i] = (ArrayList<Integer>) oldCompleteBeamOutput[candidate.parentIndex].clone();
                    if (!finished[candidate.parentIndex]) {
                        completeBeamOutput[i].add(candidate.token);
                    }
                    beamScores[i] = candidate.score;
                    finished[i] = candidate.finished;
                    inputIDs[i] = candidate.token;
                }

                CacheContainerNative oldCacheContainer = cacheContainer;
                cacheContainer = new CacheContainerNative(onnxEnv, result, WHISPER_LAYER_COUNT, beamSize, WHISPER_HEAD_COUNT, decoderInitialInputIDs.length + generatedTokenCount, WHISPER_HEAD_SIZE);
                if (oldCacheContainer != null) {
                    oldCacheContainer.close();
                }
                cacheContainer.reorder(parentIndexes);
                generatedTokenCount++;
            }
            if (!allFinished(finished)) {
                hitMaxLength = true;
            }
            int bestBeamIndex = selectBestBeam(completeBeamOutput, beamScores, beamSize);
            return new DecodeResult(completeBeamOutput[bestBeamIndex], normalizeLogProbability(beamScores[bestBeamIndex], countScoreTokens(completeBeamOutput[bestBeamIndex])), noSpeechProbability, hitMaxLength);
        } finally {
            if (result != null) {
                result.close();
            }
            if (initResult != null) {
                initResult.close();
            }
            if (initResultBatched != null) {
                initResultBatched.close();
            }
            if (cacheContainer != null) {
                cacheContainer.close();
            }
        }
    }

    private OnnxTensor createPromptTensor(int[][] decoderInitialInputIDs) throws OrtException {
        int batchSize = decoderInitialInputIDs.length;
        int promptLength = decoderInitialInputIDs[0].length;
        int[] flattenedInputIDs = new int[batchSize * promptLength];
        for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
            for (int tokenIndex = 0; tokenIndex < promptLength; tokenIndex++) {
                flattenedInputIDs[(batchIndex * promptLength) + tokenIndex] = decoderInitialInputIDs[batchIndex][tokenIndex];
            }
        }
        return TensorUtils.convertIntArrayToTensor(onnxEnv, flattenedInputIDs, new long[]{batchSize, promptLength});
    }

    private OnnxTensor createBatchedOutputEncoderTensor(OnnxTensor outputEncoder, int batchSize) throws OrtException {
        long time = System.currentTimeMillis();
        float[][] outputEncoderValue = ((float[][][]) outputEncoder.getValue())[0];
        android.util.Log.i("performance", "PhoWhisper encoder batch extract done in: " + (System.currentTimeMillis() - time) + "ms");
        time = System.currentTimeMillis();
        float[][][] outputEncoderFlatBatched = TensorUtils.batchTensor(outputEncoderValue, batchSize);
        android.util.Log.i("performance", "PhoWhisper encoder batch batching done in: " + (System.currentTimeMillis() - time) + "ms");
        time = System.currentTimeMillis();
        OnnxTensor outputEncoderBatched = TensorUtils.createFloatTensor(onnxEnv, outputEncoderFlatBatched, new long[]{batchSize, outputEncoderValue.length, outputEncoderValue[0].length}, new long[]{0});
        android.util.Log.i("performance", "PhoWhisper encoder batch creation done in: " + (System.currentTimeMillis() - time) + "ms");
        return outputEncoderBatched;
    }

    private String detokenizeOutput(ArrayList<Integer> completeOutput) throws OrtException {
        int[] sequences = completeOutput.stream().mapToInt(i -> i).toArray();
        Map detokenizerInputs = (Map) (new LinkedHashMap());
        detokenizerInputs.put("sequences", TensorUtils.createInt32Tensor(onnxEnv, sequences, new long[]{1, 1, sequences.length}));
        OrtSession.Result detokenizerOutputs = this.detokenizerSession.run(detokenizerInputs);
        Object finalTextResult = detokenizerOutputs.get(0).getValue();
        String finalText = ((String[][]) finalTextResult)[0][0];
        detokenizerOutputs.close();
        return finalText;
    }

    private DecodeResult executeLegacyWhisperGreedy(OnnxTensor outputEncoder, int[] decoderInitialInputIDs, int maxTokens) throws OrtException {
        Map<String, OnnxTensor> initInput = new HashMap<>();
        initInput.put("encoder_hidden_states", outputEncoder);
        OrtSession.Result initResult = null;
        OrtSession.Result result = null;
        OrtSession.Result oldResult = null;
        try {
            long time = System.currentTimeMillis();
            initResult = cacheInitSession.run(initInput);
            android.util.Log.i("performance", "Cache initialization done in: " + (System.currentTimeMillis() - time) + "ms");

            ArrayList<Integer> completeOutput = new ArrayList<>();
            double outputProbability = 0;
            double noSpeechProbability = 0;
            int scoreTokenCount = 0;
            boolean hitMaxLength = false;
            boolean isFirstIteration = true;
            int token = -1;
            int j = 1;

            while (token != EOS_TOKEN_ID) {
                OnnxTensor inputIDsTensor;
                if (j <= decoderInitialInputIDs.length) {
                    inputIDsTensor = TensorUtils.convertIntArrayToTensor(onnxEnv, new int[]{decoderInitialInputIDs[j - 1]});
                } else {
                    inputIDsTensor = TensorUtils.convertIntArrayToTensor(onnxEnv, new int[]{token});
                }

                Map<String, OnnxTensor> decoderInput = createDecoderInput(inputIDsTensor, result, initResult, 1, isFirstIteration);
                oldResult = result;
                result = decoderSession.run(decoderInput);
                if (oldResult != null) {
                    oldResult.close();
                }
                isFirstIteration = false;

                OnnxTensor decoderOutput = (OnnxTensor) result.get("logits").get();
                float[] outputValues = ((float[][][]) decoderOutput.getValue())[0][0];
                if (j == 1) {
                    noSpeechProbability = tokenProbability(outputValues[NO_SPEECH_TOKEN_ID], outputValues);
                }

                boolean isPromptStep = j < decoderInitialInputIDs.length;
                if (isPromptStep) {
                    token = decoderInitialInputIDs[j];
                } else {
                    applyTextLogitFilters(outputValues, j == decoderInitialInputIDs.length);
                    token = Utils.getIndexOfLargest(outputValues);
                    completeOutput.add(token);
                    outputProbability += tokenLogProbability(outputValues[token], outputValues);
                    scoreTokenCount++;
                }

                if (!isPromptStep && j - (decoderInitialInputIDs.length - 1) >= maxTokens) {
                    hitMaxLength = true;
                    token = EOS_TOKEN_ID;
                }
                j++;
            }

            return new DecodeResult(completeOutput, normalizeLogProbability(outputProbability, scoreTokenCount), noSpeechProbability, hitMaxLength);
        } finally {
            if (result != null) {
                result.close();
            }
            if (initResult != null) {
                initResult.close();
            }
        }
    }

    private DecodeResult executeCacheDecoderBeam(OnnxTensor outputEncoder, int[] decoderInitialInputIDs, int requestedBeamSize, int maxTokens) throws OrtException {
        Map<String, OnnxTensor> initInput = new HashMap<>();
        initInput.put("encoder_hidden_states", outputEncoder);
        OrtSession.Result singleInitResult = null;
        try {
            long time = System.currentTimeMillis();
            singleInitResult = cacheInitSession.run(initInput);
            android.util.Log.i("performance", "Refinement cache initialization done in: " + (System.currentTimeMillis() - time) + "ms");
            return executeCacheDecoderBeam(singleInitResult, decoderInitialInputIDs, requestedBeamSize, maxTokens);
        } finally {
            if (singleInitResult != null) {
                singleInitResult.close();
            }
        }
    }

    private DecodeResult executeCacheDecoderBeam(OrtSession.Result initResult, int[] decoderInitialInputIDs, int requestedBeamSize, int maxTokens) throws OrtException {
        int beamSize = Math.max(2, requestedBeamSize);
        android.util.Log.i("recognizer", "using Whisper beam search: " + beamSize);
        ArrayList<Integer>[] completeBeamOutput = new ArrayList[beamSize];
        for (int i = 0; i < beamSize; i++) {
            completeBeamOutput[i] = new ArrayList<>();
        }
        double[] beamScores = new double[beamSize];
        boolean[] finished = new boolean[beamSize];
        Arrays.fill(beamScores, Double.NEGATIVE_INFINITY);

        OrtSession.Result result = null;
        OrtSession.Result oldResult = null;
        OrtSession.Result initResultBatched = null;
        CacheContainerNative cacheContainer = null;
        OnnxTensor inputIDsTensor;
        int[] inputIDs = new int[beamSize];
        double noSpeechProbability = 0;
        int generatedTokenCount = 0;
        boolean hitMaxLength = false;

        try {
            boolean isFirstIteration = true;
            for (int j = 1; j <= decoderInitialInputIDs.length; j++) {
                inputIDsTensor = TensorUtils.convertIntArrayToTensor(onnxEnv, new int[]{decoderInitialInputIDs[j - 1]});
                Map<String, OnnxTensor> decoderInput = createDecoderInput(inputIDsTensor, result, initResult, 1, isFirstIteration);
                oldResult = result;
                result = decoderSession.run(decoderInput);
                if (oldResult != null) {
                    oldResult.close();
                }
                isFirstIteration = false;

                OnnxTensor decoderOutput = (OnnxTensor) result.get("logits").get();
                float[] outputValues = ((float[][][]) decoderOutput.getValue())[0][0];
                if (j == 1) {
                    noSpeechProbability = tokenProbability(outputValues[NO_SPEECH_TOKEN_ID], outputValues);
                }
                if (j < decoderInitialInputIDs.length) {
                    continue;
                }

                applyTextLogitFilters(outputValues, true);
                ArrayList<Integer> selectedTokens = new ArrayList<>();
                for (int i = 0; i < beamSize; i++) {
                    int token = Utils.getIndexOfLargest(outputValues, selectedTokens);
                    selectedTokens.add(token);
                    completeBeamOutput[i].add(token);
                    beamScores[i] = tokenLogProbability(outputValues[token], outputValues);
                    finished[i] = token == EOS_TOKEN_ID;
                    inputIDs[i] = token;
                }
                generatedTokenCount = 1;
            }

            initResultBatched = createBatchedEncoderCacheResult(initResult, beamSize);
            OrtSession.Result singleResult = result;
            result = createBatchedDecoderCacheResult(singleResult, beamSize);
            singleResult.close();

            while (!allFinished(finished) && generatedTokenCount < maxTokens) {
                inputIDsTensor = TensorUtils.convertIntArrayToTensor(onnxEnv, inputIDs, new long[]{beamSize, 1});
                Map<String, OnnxTensor> decoderInput = createDecoderInput(inputIDsTensor, result, initResultBatched, beamSize, false);
                oldResult = result;
                result = decoderSession.run(decoderInput);
                oldResult.close();

                float[][][] outputValues = (float[][][]) ((OnnxTensor) result.get("logits").get()).getValue();
                ArrayList<BeamCandidate> candidates = new ArrayList<>();
                for (int beamIndex = 0; beamIndex < beamSize; beamIndex++) {
                    if (finished[beamIndex]) {
                        candidates.add(new BeamCandidate(beamIndex, EOS_TOKEN_ID, beamScores[beamIndex], true));
                        continue;
                    }
                    float[] logits = outputValues[beamIndex][0];
                    applyTextLogitFilters(logits, false);
                    ArrayList<Integer> selectedTokens = new ArrayList<>();
                    for (int i = 0; i < beamSize; i++) {
                        int token = Utils.getIndexOfLargest(logits, selectedTokens);
                        selectedTokens.add(token);
                        candidates.add(new BeamCandidate(beamIndex, token, beamScores[beamIndex] + tokenLogProbability(logits[token], logits), token == EOS_TOKEN_ID));
                    }
                }

                BeamCandidate[] bestCandidates = selectBestCandidates(candidates, beamSize);
                ArrayList<Integer>[] oldCompleteBeamOutput = completeBeamOutput.clone();
                int[] parentIndexes = new int[beamSize];
                for (int i = 0; i < beamSize; i++) {
                    BeamCandidate candidate = bestCandidates[i];
                    parentIndexes[i] = candidate.parentIndex;
                    completeBeamOutput[i] = (ArrayList<Integer>) oldCompleteBeamOutput[candidate.parentIndex].clone();
                    if (!finished[candidate.parentIndex]) {
                        completeBeamOutput[i].add(candidate.token);
                    }
                    beamScores[i] = candidate.score;
                    finished[i] = candidate.finished;
                    inputIDs[i] = candidate.token;
                }

                CacheContainerNative oldCacheContainer = cacheContainer;
                cacheContainer = new CacheContainerNative(onnxEnv, result, WHISPER_LAYER_COUNT, beamSize, WHISPER_HEAD_COUNT, decoderInitialInputIDs.length + generatedTokenCount, WHISPER_HEAD_SIZE);
                if (oldCacheContainer != null) {
                    oldCacheContainer.close();
                }
                cacheContainer.reorder(parentIndexes);
                generatedTokenCount++;
            }
            if (!allFinished(finished)) {
                hitMaxLength = true;
            }
            int bestBeamIndex = selectBestBeam(completeBeamOutput, beamScores, beamSize);
            return new DecodeResult(completeBeamOutput[bestBeamIndex], normalizeLogProbability(beamScores[bestBeamIndex], countScoreTokens(completeBeamOutput[bestBeamIndex])), noSpeechProbability, hitMaxLength);
        } finally {
            if (result != null) {
                result.close();
            }
            if (initResultBatched != null) {
                initResultBatched.close();
            }
            if (cacheContainer != null) {
                cacheContainer.close();
            }
        }
    }

    private Map<String, OnnxTensor> createDecoderInput(OnnxTensor inputIDsTensor, OrtSession.Result decoderCache, OrtSession.Result encoderCache, int batchSize, boolean isFirstIteration) throws OrtException {
        Map<String, OnnxTensor> decoderInput = new HashMap<>();
        decoderInput.put("input_ids", inputIDsTensor);
        if (isFirstIteration) {
            long[] shape = {batchSize, WHISPER_HEAD_COUNT, 0, WHISPER_HEAD_SIZE};
            OnnxTensor decoderPastTensor = TensorUtils.createFloatTensorWithSingleValue(onnxEnv, 0, shape);
            for (int i = 0; i < WHISPER_LAYER_COUNT; i++) {
                decoderInput.put("past_key_values." + i + ".decoder.key", decoderPastTensor);
                decoderInput.put("past_key_values." + i + ".decoder.value", decoderPastTensor);
                decoderInput.put("past_key_values." + i + ".encoder.key", (OnnxTensor) encoderCache.get("present." + i + ".encoder.key").get());
                decoderInput.put("past_key_values." + i + ".encoder.value", (OnnxTensor) encoderCache.get("present." + i + ".encoder.value").get());
            }
        } else {
            for (int i = 0; i < WHISPER_LAYER_COUNT; i++) {
                decoderInput.put("past_key_values." + i + ".decoder.key", (OnnxTensor) decoderCache.get("present." + i + ".decoder.key").get());
                decoderInput.put("past_key_values." + i + ".decoder.value", (OnnxTensor) decoderCache.get("present." + i + ".decoder.value").get());
                decoderInput.put("past_key_values." + i + ".encoder.key", (OnnxTensor) encoderCache.get("present." + i + ".encoder.key").get());
                decoderInput.put("past_key_values." + i + ".encoder.value", (OnnxTensor) encoderCache.get("present." + i + ".encoder.value").get());
            }
        }
        return decoderInput;
    }

    private OrtSession.Result createBatchedEncoderCacheResult(OrtSession.Result initResult, int beamSize) throws OrtException {
        String[] names = new String[WHISPER_LAYER_COUNT * 2];
        OnnxValue[] values = new OnnxValue[WHISPER_LAYER_COUNT * 2];
        boolean[] ownedByResult = new boolean[WHISPER_LAYER_COUNT * 2];
        Arrays.fill(ownedByResult, true);
        String[] suffixes = {"key", "value"};
        int count = 0;
        for (int i = 0; i < WHISPER_LAYER_COUNT; i++) {
            for (String suffix : suffixes) {
                names[count] = "present." + i + ".encoder." + suffix;
                float[][][] keyValue = ((float[][][][]) TensorUtils.extractValue(initResult, names[count]))[0];
                values[count] = TensorUtils.createFloatTensorOptimized(onnxEnv, TensorUtils.batchTensor(keyValue, beamSize), new long[]{beamSize, keyValue.length, keyValue[0].length, keyValue[0][0].length});
                count++;
            }
        }
        return createResult(names, values, ownedByResult);
    }

    private OrtSession.Result createBatchedDecoderCacheResult(OrtSession.Result decoderResult, int beamSize) throws OrtException {
        String[] names = new String[WHISPER_LAYER_COUNT * 2];
        OnnxValue[] values = new OnnxValue[WHISPER_LAYER_COUNT * 2];
        boolean[] ownedByResult = new boolean[WHISPER_LAYER_COUNT * 2];
        Arrays.fill(ownedByResult, true);
        String[] suffixes = {"key", "value"};
        int count = 0;
        for (int i = 0; i < WHISPER_LAYER_COUNT; i++) {
            for (String suffix : suffixes) {
                names[count] = "present." + i + ".decoder." + suffix;
                float[][][] keyValue = ((float[][][][]) TensorUtils.extractValue(decoderResult, names[count]))[0];
                values[count] = TensorUtils.createFloatTensor(onnxEnv, TensorUtils.flattenFloatArrayBatched(keyValue, beamSize), new long[]{beamSize, keyValue.length, keyValue[0].length, keyValue[0][0].length});
                count++;
            }
        }
        return createResult(names, values, ownedByResult);
    }

    private OrtSession.Result createResult(String[] names, OnnxValue[] values, boolean[] ownedByResult) {
        try {
            Constructor<OrtSession.Result> constructor = OrtSession.Result.class.getDeclaredConstructor(names.getClass(), values.getClass(), ownedByResult.getClass());
            constructor.setAccessible(true);
            return constructor.newInstance(names, values, ownedByResult);
        } catch (IllegalAccessException | InstantiationException | InvocationTargetException | NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
    }

    private static BeamCandidate[] selectBestCandidates(ArrayList<BeamCandidate> candidates, int beamSize) {
        BeamCandidate[] bestCandidates = new BeamCandidate[beamSize];
        boolean[] used = new boolean[candidates.size()];
        for (int i = 0; i < beamSize; i++) {
            int bestIndex = -1;
            for (int j = 0; j < candidates.size(); j++) {
                if (!used[j] && (bestIndex == -1 || candidates.get(j).score > candidates.get(bestIndex).score)) {
                    bestIndex = j;
                }
            }
            used[bestIndex] = true;
            bestCandidates[i] = candidates.get(bestIndex);
        }
        return bestCandidates;
    }

    private static int selectBestBeam(ArrayList<Integer>[] completeBeamOutput, double[] beamScores, int beamSize) {
        int bestIndex = 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < beamSize; i++) {
            double score = normalizeLogProbability(beamScores[i], countScoreTokens(completeBeamOutput[i]));
            if (score > bestScore) {
                bestScore = score;
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    private static int countScoreTokens(ArrayList<Integer> output) {
        int count = 0;
        for (int token : output) {
            if (token != EOS_TOKEN_ID) {
                count++;
            }
        }
        return count;
    }

    private static boolean allFinished(boolean[] finished) {
        for (boolean value : finished) {
            if (!value) {
                return false;
            }
        }
        return true;
    }

    static int calculateMaxTokens(int audioLengthSamples) {
        double durationSeconds = audioLengthSamples / (double) Recorder.SAMPLE_RATE_CANDIDATES[0];
        int maxTokens = (int) Math.ceil(durationSeconds * MAX_TOKENS_PER_SECOND);
        if (maxTokens < 1) {
            maxTokens = 1;
        }
        if (maxTokens > MAX_TOKENS) {
            maxTokens = MAX_TOKENS;
        }
        return maxTokens;
    }

    static void applyTextLogitFilters(float[] logits, boolean suppressBlank) {
        suppressToken(logits, START_TOKEN_ID);
        for (int i = 0; i < LANGUAGES.length; i++) {
            suppressToken(logits, START_TOKEN_ID + i + 1);
        }
        suppressToken(logits, TRANSLATE_TOKEN_ID);
        suppressToken(logits, TRANSCRIBE_TOKEN_ID);
        suppressToken(logits, START_OF_LM_TOKEN_ID);
        suppressToken(logits, START_OF_PREV_TOKEN_ID);
        suppressToken(logits, NO_SPEECH_TOKEN_ID);
        suppressToken(logits, NO_TIMESTAMPS_TOKEN_ID);
        suppressTimestampTokens(logits);
        if (suppressBlank) {
            suppressToken(logits, BLANK_TOKEN_ID);
            suppressToken(logits, 50257);
        }
    }

    private static void suppressTimestampTokens(float[] logits) {
        for (int i = FIRST_TIMESTAMP_TOKEN_ID; i < logits.length; i++) {
            logits[i] = Float.NEGATIVE_INFINITY;
        }
    }

    private static void suppressToken(float[] logits, int tokenID) {
        if (tokenID >= 0 && tokenID < logits.length) {
            logits[tokenID] = Float.NEGATIVE_INFINITY;
        }
    }

    static double normalizeLogProbability(double outputProbability, int scoreTokenCount) {
        if (scoreTokenCount <= 0) {
            return Double.NEGATIVE_INFINITY;
        }
        return outputProbability / scoreTokenCount;
    }

    static boolean isNoSpeech(double noSpeechProbability, double averageLogProbability) {
        return noSpeechProbability > NO_SPEECH_THRESHOLD && averageLogProbability < NO_SPEECH_LOW_CONFIDENCE_THRESHOLD;
    }

    static int getEffectiveBeamSize(int requestedBeamSize, boolean hasSecondLanguage) {
        return 1;
    }

    static int getRefinementBeamSize(int requestedBeamSize, boolean hasSecondLanguage) {
        return 1;
    }

    static int selectAudioDetectedLanguageIndex(double firstProbability, double secondProbability) {
        double bestProbability = Math.max(firstProbability, secondProbability);
        if (bestProbability < AUDIO_LANGUAGE_MIN_PROBABILITY) {
            return 0;
        }
        if (Math.abs(firstProbability - secondProbability) < AUDIO_LANGUAGE_PROBABILITY_MARGIN) {
            return 0;
        }
        return firstProbability > secondProbability ? 1 : 2;
    }

    static int selectAudioDetectedLanguageIndex(float[] logits, int firstTokenID, int secondTokenID) {
        int bestLanguageTokenID = getBestLanguageTokenID(logits);
        if (bestLanguageTokenID != firstTokenID && bestLanguageTokenID != secondTokenID) {
            return 0;
        }
        return selectAudioDetectedLanguageIndex(
                languageTokenProbability(logits, firstTokenID),
                languageTokenProbability(logits, secondTokenID)
        );
    }

    static double normalizedCandidateProbability(float[] logits, int firstTokenID, int secondTokenID, int selectedTokenID) {
        if (logits == null || firstTokenID < 0 || secondTokenID < 0 || selectedTokenID < 0
                || firstTokenID >= logits.length || secondTokenID >= logits.length || selectedTokenID >= logits.length) {
            return 0;
        }
        return languageTokenProbability(logits, selectedTokenID);
    }

    static double languageTokenProbability(float[] logits, int selectedTokenID) {
        if (!isValidLogitToken(logits, selectedTokenID) || !isLanguageTokenID(selectedTokenID)) {
            return 0;
        }

        double maxLogit = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < LANGUAGES.length; i++) {
            int tokenID = START_TOKEN_ID + i + 1;
            if (isValidLogitToken(logits, tokenID) && !Float.isNaN(logits[tokenID]) && logits[tokenID] > maxLogit) {
                maxLogit = logits[tokenID];
            }
        }
        if (maxLogit == Double.NEGATIVE_INFINITY) {
            return 0;
        }

        double denominator = 0;
        for (int i = 0; i < LANGUAGES.length; i++) {
            int tokenID = START_TOKEN_ID + i + 1;
            if (isValidLogitToken(logits, tokenID) && !Float.isNaN(logits[tokenID])) {
                denominator += Math.exp(logits[tokenID] - maxLogit);
            }
        }
        if (denominator <= 0 || Double.isNaN(denominator)) {
            return 0;
        }
        return Math.exp(logits[selectedTokenID] - maxLogit) / denominator;
    }

    static int getBestLanguageTokenID(float[] logits) {
        if (logits == null) {
            return -1;
        }
        int bestTokenID = -1;
        float bestLogit = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < LANGUAGES.length; i++) {
            int tokenID = START_TOKEN_ID + i + 1;
            if (isValidLogitToken(logits, tokenID) && !Float.isNaN(logits[tokenID]) && logits[tokenID] > bestLogit) {
                bestLogit = logits[tokenID];
                bestTokenID = tokenID;
            }
        }
        return bestTokenID;
    }

    static String getLanguageCodeFromLanguageTokenID(int tokenID) {
        int languageIndex = tokenID - START_TOKEN_ID - 1;
        if (languageIndex < 0 || languageIndex >= LANGUAGES.length) {
            return "";
        }
        return LANGUAGES[languageIndex];
    }

    private static boolean isValidLogitToken(float[] logits, int tokenID) {
        return logits != null && tokenID >= 0 && tokenID < logits.length;
    }

    private static boolean isLanguageTokenID(int tokenID) {
        int languageIndex = tokenID - START_TOKEN_ID - 1;
        return languageIndex >= 0 && languageIndex < LANGUAGES.length;
    }

    static int selectAutoRefinementLanguage(boolean firstHitMaxLength, boolean firstNoSpeech, double firstScore, boolean secondHitMaxLength, boolean secondNoSpeech, double secondScore) {
        boolean firstUsable = !firstHitMaxLength && !firstNoSpeech;
        boolean secondUsable = !secondHitMaxLength && !secondNoSpeech;
        if (firstUsable && !secondUsable) {
            return 1;
        }
        if (secondUsable && !firstUsable) {
            return 2;
        }
        if (!firstUsable) {
            return 0;
        }
        return firstScore >= secondScore ? 1 : 2;
    }

    static boolean isPhoWhisperModelSetAvailable(File filesDir) {
        return new File(filesDir, PHOWHISPER_ENCODER_MODEL).exists()
                && new File(filesDir, PHOWHISPER_DECODER_INIT_MODEL).exists()
                && new File(filesDir, PHOWHISPER_DECODER_MODEL).exists();
    }

    static boolean isLegacyWhisperModelSetAvailable(File filesDir) {
        return new File(filesDir, LEGACY_WHISPER_ENCODER_MODEL).exists()
                && new File(filesDir, LEGACY_WHISPER_CACHE_INIT_MODEL).exists()
                && new File(filesDir, LEGACY_WHISPER_CACHE_INIT_BATCH_MODEL).exists()
                && new File(filesDir, LEGACY_WHISPER_DECODER_MODEL).exists();
    }

    static boolean shouldUsePhoWhisperForLanguage(String languageCode) {
        return "vi".equals(normalizeLanguageCode(languageCode));
    }

    private static String normalizeLanguageCode(String languageCode) {
        if (languageCode == null) {
            return "";
        }
        int regionSeparator = languageCode.indexOf("-");
        if (regionSeparator >= 0) {
            return languageCode.substring(0, regionSeparator);
        }
        return languageCode;
    }

    static float[] preprocessAudioForRecognition(float[] audioData, int sampleRate) {
        return normalizeAudioForRecognition(trimAudioForRecognition(audioData, sampleRate));
    }

    static float[] trimAudioForRecognition(float[] audioData, int sampleRate) {
        if (audioData == null || audioData.length == 0 || sampleRate <= 0) {
            return audioData;
        }
        int windowSize = Math.max(1, Math.round((TRIM_WINDOW_MILLIS / 1000f) * sampleRate));
        int firstActiveWindow = -1;
        int lastActiveWindow = -1;
        int windowIndex = 0;
        for (int start = 0; start < audioData.length; start += windowSize) {
            int end = Math.min(audioData.length, start + windowSize);
            if (isActiveAudioWindow(audioData, start, end)) {
                if (firstActiveWindow == -1) {
                    firstActiveWindow = windowIndex;
                }
                lastActiveWindow = windowIndex;
            }
            windowIndex++;
        }
        if (firstActiveWindow == -1) {
            return audioData;
        }
        int paddingBefore = Math.round((TRIM_PADDING_BEFORE_MILLIS / 1000f) * sampleRate);
        int paddingAfter = Math.round((TRIM_PADDING_AFTER_MILLIS / 1000f) * sampleRate);
        int startSample = Math.max(0, firstActiveWindow * windowSize - paddingBefore);
        int endSample = Math.min(audioData.length, ((lastActiveWindow + 1) * windowSize) + paddingAfter);
        if (startSample == 0 && endSample == audioData.length) {
            return audioData;
        }
        float[] trimmedAudio = new float[endSample - startSample];
        System.arraycopy(audioData, startSample, trimmedAudio, 0, trimmedAudio.length);
        return trimmedAudio;
    }

    static float[] normalizeAudioForRecognition(float[] audioData) {
        if (audioData == null || audioData.length == 0) {
            return audioData;
        }

        double mean = 0;
        for (float sample : audioData) {
            mean += sample;
        }
        mean = mean / audioData.length;

        double squareSum = 0;
        float peak = 0f;
        for (float sample : audioData) {
            float centeredSample = (float) (sample - mean);
            squareSum += centeredSample * centeredSample;
            float absSample = Math.abs(centeredSample);
            if (absSample > peak) {
                peak = absSample;
            }
        }

        double rms = Math.sqrt(squareSum / audioData.length);
        if (rms < NORMALIZATION_MIN_RMS || peak == 0f) {
            return audioData;
        }

        float gain = Math.min(NORMALIZATION_MAX_GAIN, (float) (NORMALIZATION_TARGET_RMS / rms));
        if (gain < 1f) {
            gain = 1f;
        }
        if (peak * gain > NORMALIZATION_PEAK_LIMIT) {
            gain = NORMALIZATION_PEAK_LIMIT / peak;
        }

        float[] normalizedAudio = new float[audioData.length];
        for (int i = 0; i < audioData.length; i++) {
            normalizedAudio[i] = Math.max(-NORMALIZATION_PEAK_LIMIT, Math.min(NORMALIZATION_PEAK_LIMIT, (float) ((audioData[i] - mean) * gain)));
        }
        return normalizedAudio;
    }

    private static boolean isActiveAudioWindow(float[] audioData, int start, int end) {
        float peak = 0f;
        double squareSum = 0;
        for (int i = start; i < end; i++) {
            float sample = Math.abs(audioData[i]);
            if (sample > peak) {
                peak = sample;
            }
            squareSum += sample * sample;
        }
        double rms = Math.sqrt(squareSum / Math.max(1, end - start));
        return peak > TRIM_PEAK_THRESHOLD || rms > TRIM_RMS_THRESHOLD;
    }

    static double tokenLogProbability(float selectedLogit, float[] logits) {
        return selectedLogit - Utils.logSumExpFast(logits);
    }

    static double tokenProbability(float selectedLogit, float[] logits) {
        return Math.exp(tokenLogProbability(selectedLogit, logits));
    }

    private void saveDebugAudio(float[] audioData) {
        saveDebugAudio(audioData, "debug_last_speech.wav");
    }

    private void saveDebugAudio(float[] audioData, String fileName) {
        if (!BuildConfig.DEBUG || audioData == null) {
            return;
        }
        File outputFile = new File(global.getFilesDir(), fileName);
        try (FileOutputStream outputStream = new FileOutputStream(outputFile)) {
            writeWavHeader(outputStream, audioData.length, Recorder.SAMPLE_RATE_CANDIDATES[0]);
            byte[] pcmData = new byte[audioData.length * 2];
            for (int i = 0; i < audioData.length; i++) {
                int sample = Math.round(Math.max(-1f, Math.min(1f, audioData[i])) * 32767f);
                pcmData[i * 2] = (byte) (sample & 0xff);
                pcmData[i * 2 + 1] = (byte) ((sample >> 8) & 0xff);
            }
            outputStream.write(pcmData);
            Log.i("recognizer", "saved debug audio: " + outputFile.getAbsolutePath());
        } catch (IOException e) {
            Log.w("recognizer", "Unable to save debug audio", e);
        }
    }

    private static void writeWavHeader(FileOutputStream outputStream, int sampleCount, int sampleRate) throws IOException {
        int byteRate = sampleRate * 2;
        int dataSize = sampleCount * 2;
        int fileSize = 36 + dataSize;
        outputStream.write(new byte[]{
                'R', 'I', 'F', 'F',
                (byte) (fileSize & 0xff), (byte) ((fileSize >> 8) & 0xff), (byte) ((fileSize >> 16) & 0xff), (byte) ((fileSize >> 24) & 0xff),
                'W', 'A', 'V', 'E',
                'f', 'm', 't', ' ',
                16, 0, 0, 0,
                1, 0,
                1, 0,
                (byte) (sampleRate & 0xff), (byte) ((sampleRate >> 8) & 0xff), (byte) ((sampleRate >> 16) & 0xff), (byte) ((sampleRate >> 24) & 0xff),
                (byte) (byteRate & 0xff), (byte) ((byteRate >> 8) & 0xff), (byte) ((byteRate >> 16) & 0xff), (byte) ((byteRate >> 24) & 0xff),
                2, 0,
                16, 0,
                'd', 'a', 't', 'a',
                (byte) (dataSize & 0xff), (byte) ((dataSize >> 8) & 0xff), (byte) ((dataSize >> 16) & 0xff), (byte) ((dataSize >> 24) & 0xff)
        });
    }

    public int getLanguageID(String language){
        int languageID = getLanguageTokenID(language);
        if (languageID >= 0) {
            return languageID;
        }
        Log.e("error", "Error Converting Language code " + language + " to Whisper code");
        return -1;
    }

    static int getLanguageTokenID(String language) {
        language = normalizeLanguageCode(language);
        for (int i = 0; i < LANGUAGES.length; i++) {
            if (LANGUAGES[i].equals(language)) {
                return START_TOKEN_ID + i + 1;
            }
        }
        return -1;
    }

    public void addCallback(final RecognizerListener callback) {
        callbacks.add(callback);
    }

    public void removeCallback(RecognizerListener callback) {
        callbacks.remove(callback);
    }

    public void addMultiCallback(final RecognizerMultiListener callback) {
        multiCallbacks.add(callback);
    }

    public void removeMultiCallback(RecognizerMultiListener callback) {
        multiCallbacks.remove(callback);
    }

    private void notifyResult(String text, String languageCode, double confidenceScore, boolean isFinal) {
        for (int i = 0; i < callbacks.size(); i++) {
            callbacks.get(i).onSpeechRecognizedResult(text, languageCode, confidenceScore, isFinal);
        }
    }

    private void notifyMultiResult(String text1, String languageCode1, double confidenceScore1, String text2, String languageCode2, double confidenceScore2) {
        for (int i = 0; i < multiCallbacks.size(); i++) {
            multiCallbacks.get(i).onSpeechRecognizedResult(text1, languageCode1, confidenceScore1, text2, languageCode2, confidenceScore2);
        }
    }

    private void notifyError(int[] reasons, long value) {
        for (int i = 0; i < callbacks.size(); i++) {
            callbacks.get(i).onError(reasons, value);
        }
        for (int i = 0; i < multiCallbacks.size(); i++) {
            multiCallbacks.get(i).onError(reasons, value);
        }
    }


    private static class DataContainer{
        private float[] data;
        private String languageCode;
        private String languageCode2;
        private int beamSize;
        private int refinementBeamSize;

        private DataContainer(float[] data, int beamSize, String languageCode){
            this.data = data;
            this.beamSize = beamSize;
            this.refinementBeamSize = 1;
            this.languageCode = languageCode;
        }

        private DataContainer(float[] data, int beamSize, int refinementBeamSize, String languageCode, String languageCode2){
            this.data = data;
            this.beamSize = beamSize;
            this.refinementBeamSize = refinementBeamSize;
            this.languageCode = languageCode;
            this.languageCode2 = languageCode2;
        }
    }

    private static class DecodeResult {
        private ArrayList<Integer> output;
        private double averageLogProbability;
        private double noSpeechProbability;
        private boolean hitMaxLength;

        private DecodeResult(ArrayList<Integer> output, double averageLogProbability, double noSpeechProbability, boolean hitMaxLength) {
            this.output = output;
            this.averageLogProbability = averageLogProbability;
            this.noSpeechProbability = noSpeechProbability;
            this.hitMaxLength = hitMaxLength;
        }
    }

    private static class RecognizedResult {
        private String text;
        private double averageLogProbability;

        private RecognizedResult(String text, double averageLogProbability) {
            this.text = text;
            this.averageLogProbability = averageLogProbability;
        }
    }

    private static class DetectedLanguage {
        private int index;
        private String languageCode;

        private DetectedLanguage(int index, String languageCode) {
            this.index = index;
            this.languageCode = languageCode;
        }
    }

    private static class BeamCandidate {
        private int parentIndex;
        private int token;
        private double score;
        private boolean finished;

        private BeamCandidate(int parentIndex, int token, double score, boolean finished) {
            this.parentIndex = parentIndex;
            this.token = token;
            this.score = score;
            this.finished = finished;
        }
    }
}
