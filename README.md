# Body Dimension Prediction Engine (C#)

This project provides a **C# Inference Engine (DLL)** to execute **60 individual ONNX models** trained in Python for high-precision body dimension prediction.

---

## Project Structure

| Component | Description |
| :--- | :--- |
| **ClassLibrary1.dll** | Core library containing the inference logic (ModelRunner). |
| **Python_Models/** | Directory containing ONNX model files. (Structure: [Gender]/[Model_Type]/pc1~pc60.onnx) |
| **Feature_List.json** | Defines the required order of **62 input features** (60 PCs + Stature + Weight). |

### Supported Models
"This project currently supports three specific models: ElasticNet, Weighted Ridge, and XGBoost. This limitation is due to the technical challenge of manually aligning feature lists, as these models do not include built-in feature selection mechanisms."

---

## Code Example

The input data array must have **62 elements** in the specific order defined in `Feature_List.json`.

```csharp
using AnthroInference;

ModelRunner runner = new ModelRunner("Male", "XGB");

// Input Array: [PC1...PC60 (60 elements)] + [Stature (1)] + [Weight (1)] = Total 62
List<float[]> multiPeopleData = new List<float[]>()
{
    new float[] { 
        0.12f, -0.45f, ..., 0.05f, // PC1 to PC60 (Indices 0-59)
        175.5f,                    // Stature (Index 60)
        72.0f                      // Weight (Index 61)
    }
};

foreach (float[] personData in multiPeopleData)
{
    float[] results = runner.PredictAll(personData);
    Console.WriteLine($"PC1 Prediction: {results[0]}");
}
