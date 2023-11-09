using Microsoft.ML;
using Microsoft.ML.Data;

namespace Engine
{
    public class BinaryClassification
    {
        [ColumnName("Score")]
        public float PredictedValue { get; set; }

        [ColumnName("Probability")] public float Probability { get; set; }

        [ColumnName("PredictedLabel")] public bool PredictedLabel { get; set; }
        
    }
}