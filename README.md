Dynamic DataView and Pipelines for ML.NET.

Allow inherited classes to be used as DataView by casting to concrete classes at runtime, allows IDataView to use object and dynamic types which can be cast back to .GetType() at runtime. 

Simplifies code by allowing many data types to call the same training method.
