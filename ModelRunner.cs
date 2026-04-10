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
        // List that will contain 60 sessions
        private List<InferenceSession> _sessions = new List<InferenceSession>();

        public ModelRunner(string gender, string modelType, string modelRootPath = "Python_Models")
        {
            // Example: Python_Models/Male/Model_Name/
            string folderPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, modelRootPath, gender, modelType);

            if (!Directory.Exists(folderPath))
                throw new DirectoryNotFoundException($"Cannot find Folder: {folderPath}");

            //  Load pc1.onnx to pc60.onnx with keeping order

            for (int i = 1; i <= 60; i++)
            {
                string filePath = Path.Combine(folderPath, $"pc{i}.onnx");
                if (File.Exists(filePath))
                {
                    _sessions.Add(new InferenceSession(filePath));
                }
            }
        }

        // run all 60 models and save 60 results from each model
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
                    // Each model has one result (e.g pc1,onnx -> predicted pc1)
                    finalResults[i] = results.First().AsEnumerable<float>().First();
                }
            }

            return finalResults; // Return arry of 60 pc prediction (number of pcs can vary based on input)
        }
    }
}