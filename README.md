Body Dimension Prediction Engine (C#)
This project provides a C# Inference Engine (DLL) to execute 60 individual ONNX models trained in Python for body dimension prediction.

Project Structure
ClassLibrary1.dll: Core library containing the inference logic (ModelRunner).

Python_Models/: Directory containing ONNX model files.

Structure: [Gender]/[Model_Type]/pc1.onnx to pc60.onnx.

Feature_List.txt: A reference file defining the required order of input features (Principal Components).

Getting Started
1. Prerequisites
.NET Framework 4.7.2 or higher

NuGet Package: Microsoft.ML.OnnxRuntime

2. Setup
Add ClassLibrary1.dll as a reference in your C# project.

Ensure the Python_Models folder is located in the same directory as your executable file (/bin/Debug or /bin/Release).

Code Example
This example demonstrates how to initialize the engine and process multiple sets of PC (Principal Component) data using a loop.

C#
using AnthroInference; // Namespace defined in your DLL

// 1. Initialize Engine
// Specify Gender ("Male" or "Female") and Model Type ("Anthro" or "PCA")
ModelRunner runner = new ModelRunner("Male", "Anthro");

// 2. Prepare Input Data (PC values: PC1, PC2, PC3...)
// Input values must strictly follow the order defined in Feature_List.txt
List<float[]> multiPeopleData = new List<float[]>()
{
    // Example: { PC1, PC2, PC3, ... }
    new float[] { 0.12f, -0.45f, 1.02f, ... }, // Person 1
    new float[] { -0.88f, 0.22f, -0.15f, ... }, // Person 2
    new float[] { 1.34f, -1.10f, 0.05f, ... }   // Person 3
};

// 3. Execute Predictions
foreach (float[] personPCData in multiPeopleData)
{
    // PredictAll runs 60 models and returns an array of 60 predicted values
    float[] results = runner.PredictAll(personPCData);
    
    // results[0] is the prediction for the first target, results[59] for the 60th
    Console.WriteLine($"Prediction Complete. PC1 Result: {results[0]}, PC60 Result: {results[59]}");
}
Notes
Input Order: The model is highly sensitive to the order of Principal Components. Refer to Feature_List.txt to map your PC values to the correct array indices.

Runtime Path: The ClassLibrary1.dll expects the Python_Models folder to be in the application's base directory. Please maintain the provided folder structure.
