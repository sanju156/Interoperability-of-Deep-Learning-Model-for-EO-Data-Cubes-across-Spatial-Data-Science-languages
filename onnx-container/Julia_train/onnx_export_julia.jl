using Flux
using ONNXNaiveNASflux
using BSON

BSON.@load "/tempcnn_julia_model.bson" model_cpu
model = model_cpu 

try
    ONNXNaiveNASflux.save("/tempcnn_julia_model.onnx", model, (45, 13, :N))
    println("Model successfully exported to tempcnn_julia_model.onnx")

catch e
    println("Method failed: ", e)
end