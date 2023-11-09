Dynamic Schema, DataView and Pipelines for ML.NET.

Allow inherited classes to be used as DataView by casting to concrete classes at runtime, allowing IDataView to use RuntTime object types. 

No need to use json or csv or textfiles as a DataView, thus we have a RunTime dynamic DataView of a live IEnumerable of objects, and can adjust our DataView to any new Type that is created which requires its own Context.Model

Simplifies code by allowing many data types to call the same training method, engine method and prediction method simply by inheriting the same interface (even an empty interface)

