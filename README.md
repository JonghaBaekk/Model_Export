# Body Dimension Prediction Engine (C#)

This project provides a **C# Inference Engine (DLL)** to execute **60 individual ONNX models** trained in Python for high-precision body dimension prediction.

---

## Project Structure

| Component | Description |
| :--- | :--- |
| **ClassLibrary1.dll** | Core library containing the inference logic (ModelRunner). |
| **Python_Models/** | Directory containing ONNX model files. (Structure: [Gender]/[Model_Type]/pc1~pc60.onnx) |
| **Feature_List.txt** | A reference file defining the required order of input features (Principal Components). |

---

## Getting Started

### 1. Prerequisites
* **.NET Framework 4.7.2** or higher
* **NuGet Package**: `Microsoft.ML.OnnxRuntime`

### 2. Setup
1. **Add Reference**: Add `ClassLibrary1.dll` as a reference in your C# project.
2. **Asset Deployment**: Ensure the `Python_Models` folder is located in the same directory as your executable file (e.g., `/bin/Debug` or `/bin/Release`).

---

## Code Example

The following example demonstrates how to initialize the engine and process multiple sets of **PC (Principal Component)** data using a loop.

```csharp
using AnthroInference; // Namespace defined in your DLL

// 1. Initialize Engine
// Specify Gender ("Male" or "Female") and Model Type ("Anthro" or "PCA")
ModelRunner runner = new ModelRunner("Male", "Anthro");

// 2. Prepare Input Data (PC values: PC1, PC2, PC3...)
// NOTE: Input values must strictly follow the order defined in Feature_List.txt
List<float[]> multiPeopleData = new List<float[]>()
{
    // Format: { PC1, PC2, PC3, ... }
    new float[] { 0.12f, -0.45f, 1.02f, ... }, // Person 1
    new float[] { -0.88f, 0.22f, -0.15f, ... }, // Person 2
    new float[] { 1.34f, -1.10f, 0.05f, ... }   // Person 3
};

// 3. Execute Predictions
foreach (float[] personPCData in multiPeopleData)
{
    // PredictAll runs 60 models and returns an array of 60 predicted values
    float[] results = runner.PredictAll(personPCData);
    
    // results[0] is the prediction for the first target (PC1)
    // results[59] is the prediction for the 60th target (PC60)
    Console.WriteLine($"Prediction Complete.");
    Console.WriteLine($"-> PC1 Result: {results[0]} / PC60 Result: {results[59]}");
}
