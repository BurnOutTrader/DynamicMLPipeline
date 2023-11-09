using Microsoft.ML;
using Microsoft.ML.Data;

namespace Foods
{
    
    public interface IFood
    {
        bool Answer { get; set; }

        abstract void SubmitAnswer(float random);
    }
    
    public class FoodType1 : IFood
    {
        public float Value1 { get; set; }
        
        public float Random { get; set; }
        
        public bool Answer { get; set; }
        
        public FoodType1(float value1, float value2)
        {
            Value1 = value1;
            Random = value2;
        }
        
        public void SubmitAnswer(float random)
        {
            Answer = Value1 + Random + random > 50;
        }
    }
    
    public class FoodType2 : IFood
    {
        public float Value1 { get; set; }
        
        public float Random { get; set; }
        
        public float Value3 { get; set; }
        
        public bool Answer { get; set; }
        
        public FoodType2(float value1, float value2, float value3)
        {
            Value1 = value1;
            Random = value2;
            Value3 = value3;
        }
        
        public void SubmitAnswer(float random)
        {
            Answer = Value1 + Random + Value3 + random > 50;
        }
    }
}

