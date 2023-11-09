Dynamic DataView and Pipelines for ML.NET.

Allow inherited classes to be used as DataView by casting to concrete classes at runtime, allows IDataView to use object and dynamic types which can be cast back to .GetType() at runtime. 

No need to use json or csv or textfiles as a DataView, thus we have a RunTime dynamic DataView of a live IEnumerable of objects, and can adjust out DataView to any new Type that is created which requires its own Context.Model

Simplifies code by allowing many data types to call the same training method.

Note *There is no check for Array lengths to make sure they are all equal, this will have to be specific to use case.*
