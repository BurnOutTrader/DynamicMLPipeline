using System.Reflection;
using System.Runtime.Serialization;
using Foods;
using Microsoft.ML;
using Microsoft.ML.Data;
using Newtonsoft.Json;

namespace Engine
{
    public static class MainEngine
    {
        public static Dictionary<Type, List<IFood>> GenerateFakeData()
        {
            var foods = new Dictionary<Type, List<IFood>>();
            //Generate random training data for the prediction example
            for (var i = 0; i < 6000; i++)
            {
                float random = new Random().Next(0, 10);
                var food = new FoodType1(i, random/ 2);
                var food2 = new FoodType2(i, i + random /2, random);
                food.SubmitAnswer(random);
                food2.SubmitAnswer(random);
                
                if (!foods.ContainsKey(food.GetType()))
                {
                    foods.Add(food.GetType(), new List<IFood>());
                    foods.Add(food2.GetType(), new List<IFood>());
                }
                foods[food.GetType()].Add(food);
                foods[food2.GetType()].Add(food2);
            }

            return foods;
        }
        
        public static Dictionary<Type, MLContext> Context = new();
        public static Dictionary<Type, ITransformer> Model = new();
        public static void Main()
        {
            var Foods = GenerateFakeData();

            //Build the DataView dynamically based on the concrete clas properties
            foreach (var (type, foodList) in Foods )
            {
                var context = new MLContext();

                // Check if the food collection is empty or null.
                if (foodList == null || !foodList.Any())
                    throw new InvalidOperationException("The food collection is empty or null.");

                // Group food items by their concrete types.
                var groupedByType = foodList.GroupBy(f => f.GetType()).ToList();
                

                foreach (var group in groupedByType)
                {
                    var concreteType = group.Key;
                    var schemaDef = SchemaFactory.Build(group.Key, group.First());
                    
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
                    
                    //the food and conext are stored by the type of food they consume as a key
                    Context.Add(type, context);
                    Model.Add(type, model);
                }
            }
            
            Console.WriteLine("Models Trained proceeding to Test Predictions");
            
            //generate test foods. returns a dictionary where the food type is the key. this could be used on creating foods to keep them stored.
            var testFoods = GenerateFakeData();
            foreach (var (type, context) in Context)
            {
                DynamicPredict(context, type, testFoods[type]);
            }
        }

        /// <summary>
        /// We can still save our prediction engine as an generic object so we dont have to create it again for each prediction
        /// </summary>
        public static Dictionary<Type, object> PredictionEngine = new();
        
        /// <summary>
        /// Here we are overriding the ML.Net tendency to only recognise base class properties by invoking a dynamic/reflective instance of the predict method
        /// (sorry I am not a professional so my language is probably inaccurate.
        /// </summary>
        private static void DynamicPredict(MLContext context, Type type, List<IFood> testFoods)
        {
            var model = Model[type];
            var foodType = type; // Concrete type derived from IFood

            //The real trick is this combined with the schema factory
            // Get the CreatePredictionEngine method info with specific parameter types
            MethodInfo createEngineMethod = context.Model.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .Single(m => m.Name == "CreatePredictionEngine" 
                             && m.GetParameters().Length == 4 // Assuming the method has 4 parameters
                             && m.GetParameters()[0].ParameterType == typeof(ITransformer)
                             && m.GetParameters()[1].ParameterType == typeof(bool)
                             && m.GetParameters()[2].ParameterType == typeof(SchemaDefinition)
                             && m.GetParameters()[3].ParameterType == typeof(SchemaDefinition))
                .MakeGenericMethod(foodType, typeof(BinaryClassification)); // Assuming BinaryClassification is your output class

            
            if (!PredictionEngine.ContainsKey(type))
            {
                //dynamically create the prediction engine and save it as a generic object to the dictionary, by the type of food it consumes.(key)
                var predictionEngine = createEngineMethod.Invoke(context.Model, new object[] { model, false, null, null });
                PredictionEngine.Add(type, predictionEngine);
            }
            
            // Get the method info for the Predict function of the prediction engine
            MethodInfo predictMethod = PredictionEngine[foodType].GetType().GetMethod("Predict", new[] { foodType });
            
            
            // Make predictions on each food item
            foreach (var food in testFoods)
            {
                // Predict dynamically
                var prediction = predictMethod.Invoke(PredictionEngine[foodType], new[] { food });
                
                Console.WriteLine($"Prediction for {foodType.Name}: {(prediction as BinaryClassification).PredictedLabel}");
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
        

        /// <summary>
        /// Filters out columns that we dont want to include in our schema when making predictions
        /// </summary>
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


        public static List<IFood> TestData = new();
        /// <summary>
        /// This is a method i use with a while loop to train from serialised data, to save memory.
        /// </summary>
        /// <returns></returns>
        public static List<IFood> LoadNextTrain(string foodFolder)
        {
            var files = Directory.GetFiles(foodFolder);
            if (!files.Any())
            {
                return null;
            }
            var foods = new List<IFood>();

            foreach (var file in files)
            {
                var settings = new JsonSerializerSettings
                {
                    TypeNameHandling = TypeNameHandling.All
                };
                var foodList = JsonConvert.DeserializeObject<List<IFood>>(file, settings);

                foods.AddRange(foodList);
            }

            if (foods == null || foods.Count == 0)
            {
                return null;
            }
            
            var rng = new Random();
            var shuffledFoodList = foods.OrderBy(a => rng.Next()).ToList();

            var splitIndex = (int)(shuffledFoodList.Count * 0.95); 

            var trainingData = shuffledFoodList.Take(splitIndex);
            var testData = shuffledFoodList.Skip(splitIndex);
            TestData.AddRange(testData);
            
            return trainingData.ToList();
        }
    }
}

