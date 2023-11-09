using System.Reflection;
using System.Runtime.Serialization;
using Foods;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Engine
{
    public static class SchemaFactory
    {
        public static SchemaDefinition Build(Type foodType, IFood sampleFood)
        {
            var schemaDef = SchemaDefinition.Create(foodType);

            var allColumnNames = GetAllColumnNames(foodType);

            foreach (var property in foodType.GetProperties())
            {
                // Only include properties that are part of the feature column names or the label.
                if (!allColumnNames.Contains(property.Name))
                    continue;

                if (property.PropertyType == typeof(float[]))
                {
                    var vectorSize = ((float[])property.GetValue(sampleFood)!).Length;
                    schemaDef[property.Name].ColumnType =
                        new VectorDataViewType(NumberDataViewType.Single, vectorSize);
                }
                else if (property.PropertyType == typeof(float))
                {
                    schemaDef[property.Name].ColumnType = NumberDataViewType.Single;
                }
                else if (property.PropertyType == typeof(bool))
                {
                    schemaDef[property.Name].ColumnType = BooleanDataViewType.Instance;
                }
                else if (property.PropertyType == typeof(decimal))
                {
                    schemaDef[property.Name].ColumnType = BooleanDataViewType.Instance;
                }
            }

            return schemaDef;
        }

        public static string[] GetAllColumnNames(Type foodType)
        {
            var bindingFlags = BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly;
            var featureColumnNames = foodType
                .GetProperties(bindingFlags)
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