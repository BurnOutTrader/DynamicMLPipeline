using System.Reflection;
using System.Runtime.Serialization;
using Foods;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Engine
{
    public static class MainEngine
    {
        public static Dictionary<Type, MLContext> Context = new();
        public static Dictionary<Type, ITransformer> Model = new();
        public static void Main()
        {
            var Foods = new Dictionary<Type, List<IFood>>();
            
            //Generate random training data for the example
            for (var i = 0; i < 3000; i++)
            {
                float random = new Random().Next(0, 10);
                var food = new FoodType1(i, random/ 2);
                var food2 = new FoodType2(i, i + random /2, random);
                food.SubmitAnswer(random);
                food2.SubmitAnswer(random);
                
                
                if (!Foods.ContainsKey(food.GetType()))
                {
                    Foods.Add(food.GetType(), new List<IFood>());
                    Foods.Add(food2.GetType(), new List<IFood>());
                }
                Foods[food.GetType()].Add(food);
                Foods[food2.GetType()].Add(food2);
            }

            //Build the DataView dynamically based on the concrete clas properties
            foreach (var food in Foods )
            {
                var context = new MLContext();

                // Check if the food collection is empty or null.
                if (food.Value == null || !food.Value.Any())
                    throw new InvalidOperationException("The food collection is empty or null.");

                // Group food items by their concrete types.
                var groupedByType = food.Value.GroupBy(f => f.GetType()).ToList();
                

                foreach (var group in groupedByType)
                {
                    var DataView = SchemaFactory.Build(group.Key, group.First());
                    var concreteType = group.Key;
                    var schemaDef = SchemaDefinition.Create(concreteType);
                    
                    //The column names where the values are known before predictions
                    var featureColumnNames = FeatureColumnFiltering.GetFeautureColumnNames(concreteType);
                    

                    // Load the data into an IDataView, applying the schema definition.
                    // Cast each element to the concrete type.
                    var dataView = LoadDataFromEnumerableDynamic(context, concreteType, group, schemaDef);
                    
                    var dataSplit = context.Data.TrainTestSplit(dataView, testFraction: 0.2);
                    var trainData = dataSplit.TrainSet;
                    var testData = dataSplit.TestSet;
                    
                    //Build the pipeline
                    var options = new Microsoft.ML.Trainers.LbfgsLogisticRegressionBinaryTrainer.Options
                    {
                        LabelColumnName = "Answer",
                        FeatureColumnName = "Features",
                        OptimizationTolerance = 1e-4f,
                    };
                    var pipeline3 = context.Transforms.Concatenate("Features", featureColumnNames)
                        .Append(context.Transforms.NormalizeMinMax("Features"))
                        .Append(context.BinaryClassification.Trainers.LbfgsLogisticRegression(options));

                    var model = pipeline3!.Fit(trainData);
                
                    var predictions = model.Transform(testData);
                    var metrics = context.BinaryClassification.Evaluate(predictions, "Answer");
                    
                    Console.WriteLine("Checking all properties were added to the schema for the concrete type");
                    foreach (PropertyInfo property in concreteType.GetProperties())
                    {
                        
                        // Print the name and value of the property
                        Console.WriteLine($"{property.Name}");
                    }

                    Console.WriteLine($"MicroAccuracy: {metrics.Accuracy:F2}, LogLoss: {metrics.LogLoss:F2}, LogLossReduction: {metrics.LogLossReduction:F2}");
                    
                    Context.Add(food.Key, context);
                    Model.Add(food.Key, model);
                }
            }
            
            Console.WriteLine("Models Trained proceeding to Test Predictions");
            
            foreach (var kvp in Context)
            {
                var mlContext = kvp.Value;
                var model = Model[kvp.Key];
                var foodType = kvp.Key; // Concrete type derived from IFood

                // Get the CreatePredictionEngine method info with specific parameter types
                MethodInfo createEngineMethod = mlContext.Model.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance)
                    .Single(m => m.Name == "CreatePredictionEngine" 
                                 && m.GetParameters().Length == 4 // Assuming the method has 4 parameters
                                 && m.GetParameters()[0].ParameterType == typeof(ITransformer)
                                 && m.GetParameters()[1].ParameterType == typeof(bool)
                                 && m.GetParameters()[2].ParameterType == typeof(SchemaDefinition)
                                 && m.GetParameters()[3].ParameterType == typeof(SchemaDefinition))
                    .MakeGenericMethod(foodType, typeof(BinaryClassification)); // Assuming BinaryClassification is your output class

                // Create the prediction engine dynamically
                var predictionEngine = createEngineMethod.Invoke(mlContext.Model, new object[] { model, false, null, null });

                // Get the method info for the Predict function of the prediction engine
                MethodInfo predictMethod = predictionEngine.GetType().GetMethod("Predict", new[] { foodType });

                // Make predictions on each food item
                foreach (var food in Foods[foodType])
                {
                    // Predict dynamically
                    var prediction = predictMethod.Invoke(predictionEngine, new[] { food });
                    Console.WriteLine($"Prediction for {foodType.Name}: {(prediction as BinaryClassification).PredictedLabel}");
                }
            }
        }

        public static IDataView LoadDataFromEnumerableDynamic(MLContext context, Type dataType, IEnumerable<IFood> data, SchemaDefinition schemaDef)
        {
            // Get the method info for LoadFromEnumerable<>
            var loadDataMethodInfo = typeof(DataOperationsCatalog).GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .First(mi => mi.Name == "LoadFromEnumerable" && mi.GetParameters().Length == 2 
                                                             && mi.GetParameters()[1].ParameterType == typeof(SchemaDefinition));

            // Make the generic method with the specific data type
            var genericLoadDataMethodInfo = loadDataMethodInfo.MakeGenericMethod(dataType);

            // Convert the data to the correct type before calling the generic method
            var castedData = Cast(data, dataType);

            // Invoke the method with the casted data and the schema definition
            return (IDataView)genericLoadDataMethodInfo.Invoke(context.Data, new object[] { castedData, schemaDef });
        }

        // Helper method to cast the elements of an IEnumerable to the specified type
        private static IEnumerable<object> Cast(IEnumerable<IFood> source, Type targetType)
        {
            var castMethod = typeof(Enumerable).GetMethod(nameof(Enumerable.Cast)).MakeGenericMethod(targetType);
            return (IEnumerable<object>)castMethod.Invoke(null, new object[] { source });
        }


        // Helper method to cast the elements of an IGrouping to the specified type
        private static IEnumerable<object> CastGroup(IGrouping<Type, object> group, Type targetType)
        {
            // Use reflection to call Enumerable.Cast<T> on the elements of the group
            var castMethod = typeof(Enumerable).GetMethod(nameof(Enumerable.Cast)).MakeGenericMethod(targetType);
            return (IEnumerable<object>)castMethod.Invoke(null, new object[] { group.AsEnumerable() });
        }

        
        public static class FeatureColumnFiltering
        {
            //We are trying to predict "Answer" before we know "Random" So we ignore these columns as they will not be present before predictions
            public static string[] GetFeautureColumnNames(Type foodType)
            {
                var bindingFlags = BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly;
                var featureColumnNames = foodType
                    .GetProperties(bindingFlags)
                    .Where(p =>
                        p.Name != "Answer" &&
                        (p.PropertyType == typeof(float) || p.PropertyType == typeof(float[]))
                    )
                    .Select(p =>
                    {
                        // Get all custom attributes of the property and find the ColumnNameAttribute
                        var customAttribute = p.GetCustomAttributes(false)
                            .FirstOrDefault(attr => attr.GetType().Name == "ColumnNameAttribute");
                        if (customAttribute != null)
                        {
                            // Get the Name property from the ColumnNameAttribute
                            var nameProperty = customAttribute.GetType().GetProperty("Name");
                            if (nameProperty != null)
                            {
                                return (string)nameProperty.GetValue(customAttribute);
                            }
                        }

                        // If the attribute is not found or it does not have a Name property, return the property name itself
                        return p.Name;
                    })
                    .ToArray();

                return featureColumnNames;
            }
        }
    }
}

