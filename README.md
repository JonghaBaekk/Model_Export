# Body Dimension Prediction Engine (C#)

This project provides a C# Inference Engine (DLL) to execute 60 individual ONNX models trained in Python for high-precision body dimension prediction. It combines Principal Component (PC) data with basic physical metrics (Stature, Weight) to provide accurate estimations.

---

## Project Structure

| Component | Description |
| :--- | :--- |
| **ClassLibrary1.dll** | Core library containing the inference logic (ModelRunner). |
| **Python_Models/** | Directory containing ONNX model files. (Structure: [Gender]/[Model_Type]/pc1~pc60.onnx) |
| **Feature_List.json** | Defines the required order of 62 input features (60 PCs + Stature + Weight). |

### Supported Models
Currently, this project supports three specific models: **ElasticNet, Weighted Ridge, and XGBoost (XGB)**.

**Reason for Limited Support:**
The support is focused on these three models because they maintain a fixed, consistent feature set (60 PCs + Stature + Weight). Other models from the original repository, such as Lasso or Stepwise Regression, involve dynamic feature selection where the specific variables used can vary. Synchronizing these dynamic feature lists between the Python training environment and the C# inference engine without a complex internal mapping logic is technically challenging; therefore, this engine is optimized for models with a predefined 62-input structure.

*Reference for training logic:* [UMTRI_Dimension_Prediction](https://github.com/JonghaBaekk/UMTRI_Dimension_Prediction.git)

---

## Model Deployment Requirements

For the engine to function correctly, the folder structure must be strictly maintained:
- **Root Path**: Python_Models/[Gender]/[Model_Name]/
- **Files**: Each model folder must contain exactly 60 files named pc1.onnx through pc60.onnx.
- **Example Path**: Python_Models/Male/XGB/pc1.onnx ... pc60.onnx

---

## Code Example

The input array must contain exactly 62 elements in the specific order defined in Feature_List.json.

```csharp
using AnthroInference; // Namespace defined in ClassLibrary1.dll

// 1. Initialize Engine
// Parameters: Gender ("Male"/"Female"), Model Type ("Elastic", "Weighted_Ridge", or "XGB")
ModelRunner runner = new ModelRunner("Male", "XGB");

// 2. Prepare Input Data
// Format: [PC1...PC60 (Index 0-59)] + [Stature (Index 60)] + [Weight (Index 61)]
List<float[]> multiPeopleData = new List<float[]>()
{
    new float[] { 
        0.12f, -0.45f, 1.02f, ..., // 60 Principal Components (Indices 0-59)
        175.5f,                    // Stature (Index 60)
        72.0f                      // Weight (Index 61)
    }
};

// 3. Execute Predictions
foreach (float[] personData in multiPeopleData)
{
    // PredictAll executes all 60 models and returns an array of 60 predicted values
    float[] results = runner.PredictAll(personData);
    
    Console.WriteLine("Prediction Complete.");
    Console.WriteLine($"-> PC1 Result: {results[0]} / PC60 Result: {results[59]}");
}
