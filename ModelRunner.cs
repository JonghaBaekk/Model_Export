using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AnthroInference
{
    public class ModelRunner
    {
        // 60개의 세션을 담을 리스트
        private List<InferenceSession> _sessions = new List<InferenceSession>();

        public ModelRunner(string gender, string modelType, string modelRootPath = "Python_Models")
        {
            // 예: Python_Models/Male/Anthro/
            string folderPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, modelRootPath, gender, modelType);

            if (!Directory.Exists(folderPath))
                throw new DirectoryNotFoundException($"폴더를 찾을 수 없습니다: {folderPath}");

            // pc1.onnx부터 pc60.onnx까지 순서대로 불러오기
            for (int i = 1; i <= 60; i++)
            {
                string filePath = Path.Combine(folderPath, $"pc{i}.onnx");
                if (File.Exists(filePath))
                {
                    _sessions.Add(new InferenceSession(filePath));
                }
            }
        }

        // 60개 모델을 모두 돌려 60개의 결과값을 모으는 함수
        public float[] PredictAll(float[] inputData)
        {
            float[] finalResults = new float[_sessions.Count];

            for (int i = 0; i < _sessions.Count; i++)
            {
                var container = new List<NamedOnnxValue>();
                var tensor = new DenseTensor<float>(inputData, new[] { 1, inputData.Length });
                container.Add(NamedOnnxValue.CreateFromTensor("float_input", tensor));

                using (var results = _sessions[i].Run(container))
                {
                    // 각 모델은 결과값 1개를 내뱉음
                    finalResults[i] = results.First().AsEnumerable<float>().First();
                }
            }

            return finalResults; // 최종적으로 60개의 예측값이 담긴 배열 반환
        }
    }
}