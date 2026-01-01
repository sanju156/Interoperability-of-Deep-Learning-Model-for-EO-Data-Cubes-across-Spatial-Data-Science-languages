using NPZ, CSV, DataFrames, Flux, MLBase, OneHotArrays, CUDA, ProgressMeter, BSON, Statistics
using Flux: Chain, Conv, Dense, BatchNorm, relu, Dropout, DataLoader, onecold, params
using LinearAlgebra
using Printf
const flux_flatten = Flux.flatten
# === Config ===
DATA_PATH = "path to the numpy files of the training regions/data"
REGIONS = ["frh01", "frh02", "frh03"]
BATCH_SIZE = 1024
EPOCHS = 11
LEARNING_RATE = 2.38e-4
WEIGHT_DECAY = 5.18e-5
DROPOUT_RATE = 0.182039
N_CLASSES = 9
USE_GPU = CUDA.has_cuda()
METRICS_CSV = "/train_metrics_julia.csv"

isfile(METRICS_CSV) && rm(METRICS_CSV; force=true)

# === Data Loading ===
X_list = Array{Float32,3}[]
y_list = Int32[]

for region in REGIONS
    X_region = npzread(joinpath(DATA_PATH, "X_$(region).npy"))
    y_region = npzread(joinpath(DATA_PATH, "y_true_$(region).npy"))

    X_region = permutedims(Float32.(X_region), (2, 3, 1))
    y_region = Int32.(y_region .+ 1)

    push!(X_list, X_region)
    append!(y_list, y_region)
end

X_input = cat(X_list..., dims=3)
y_oh = onehotbatch(y_list, 1:N_CLASSES)

println("Training samples: ", size(X_input, 3))
println("Input shape: ", size(X_input), " | Label shape: ", size(y_oh))

if USE_GPU
    X_input, y_oh = cu(X_input), cu(y_oh)
end

# Dropout that works on both CPU & GPU 
function device_dropout(p::Float64)
    if CUDA.has_cuda()
        return Dropout(p, rng=CUDA.default_rng())  
    else
        return Dropout(p)  
    end
end

# Conv1D + BatchNorm + ReLU + Dropout Block 
function Conv1D_BN_ReLU_Drop(in_ch, out_ch, ks)
    return Chain(
        Conv((ks,), in_ch => out_ch, pad=(ks ÷ 2)),
        BatchNorm(out_ch),
        relu,
        device_dropout(DROPOUT_RATE)
    )
end

# === Model Definition ===
model = Chain(
    # Feature extraction blocks
    Conv1D_BN_ReLU_Drop(13, 128, 7),
    Conv1D_BN_ReLU_Drop(128, 128, 7),
    Conv1D_BN_ReLU_Drop(128, 128, 7),

    # Flatten
    flux_flatten,

    # Dense + BatchNorm + ReLU + Dropout block
    Chain(
        Dense(128 * 45, 512),
        BatchNorm(512),
        relu,
        device_dropout(DROPOUT_RATE),
    ),

    # Final classification layer
    Dense(512, N_CLASSES)
)

# Move to GPU if available
USE_GPU && (model = fmap(cu, model))

# === Training Setup ===
train_loader = DataLoader((X_input, y_oh), batchsize=BATCH_SIZE, shuffle=true)
loss_fn(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)
opt = Flux.AdamW(LEARNING_RATE, (0.9, 0.999), WEIGHT_DECAY)
opt_state = Flux.setup(opt, model)

CSV.write(METRICS_CSV, DataFrame(
    Epoch=Int[], Loss=Float64[], Accuracy=Float64[], Kappa=Float64[],
    F1_micro=Float64[], F1_macro=Float64[], F1_weighted=Float64[],
    Precision_micro=Float64[], Precision_macro=Float64[], Precision_weighted=Float64[],
    Recall_micro=Float64[], Recall_macro=Float64[], Recall_weighted=Float64[],
    Iterations_per_sec=Float64[], Epoch_time_sec=Float64[]
))

# === Training Loop ===
Flux.trainmode!(model)

for epoch in 1:EPOCHS
    total_loss = 0.0
    y_preds, y_true = Int[], Int[]
    prog = Progress(length(train_loader), desc="Epoch $epoch")
    t_start = time()

    for (x_batch, y_batch) in train_loader
        loss, grads = Flux.withgradient(model) do m
            ŷ = m(x_batch)
            loss_fn(ŷ, y_batch)
        end
        Flux.update!(opt_state, model, grads[1])
        total_loss += loss

        logits = model(x_batch)
        pred = onecold(cpu(logits), 1:N_CLASSES)
        truth = onecold(cpu(y_batch), 1:N_CLASSES)

        append!(y_preds, pred)
        append!(y_true, truth)

        next!(prog)
    end

    duration = time() - t_start
    iters_per_sec = length(train_loader) / duration
    
    # Create confusion matrix 
    conf_matrix = zeros(Int, N_CLASSES, N_CLASSES)
    for i in 1:length(y_true)
        conf_matrix[y_true[i], y_preds[i]] += 1
    end
    
    # Calculate per-class metrics
    precision_pc = Float64[]
    recall_pc = Float64[]
    f1_pc = Float64[]
    
    for class in 1:N_CLASSES
        # True positives, false positives, false negatives
        tp = conf_matrix[class, class]
        fp = sum(conf_matrix[:, class]) - tp
        fn = sum(conf_matrix[class, :]) - tp
        
        # Precision and recall
        prec = tp == 0 && fp == 0 ? 0.0 : tp / (tp + fp)
        rec = tp == 0 && fn == 0 ? 0.0 : tp / (tp + fn)
        f1 = prec + rec == 0 ? 0.0 : 2 * prec * rec / (prec + rec)
        
        push!(precision_pc, prec)
        push!(recall_pc, rec)
        push!(f1_pc, f1)
    end
    
    # Macro averages
    precision_macro = mean(precision_pc)
    recall_macro = mean(recall_pc)
    f1_macro = mean(f1_pc)
    
    # Micro averages (same as accuracy for multiclass)
    accuracy = mean(y_preds .== y_true)
    precision_micro = accuracy
    recall_micro = accuracy
    f1_micro = accuracy
    
    # Weighted averages
    class_support = [sum(y_true .== c) for c in 1:N_CLASSES]
    total_support = sum(class_support)
    weights = class_support ./ total_support
    
    precision_weighted = sum(precision_pc .* weights)
    recall_weighted = sum(recall_pc .* weights)
    f1_weighted = sum(f1_pc .* weights)
    
    # Cohen's Kappa
    observed_accuracy = accuracy
    expected_accuracy = sum((sum(conf_matrix[i, :]) * sum(conf_matrix[:, i])) for i in 1:N_CLASSES) / (total_support^2)
    kappa = expected_accuracy == 1.0 ? 0.0 : (observed_accuracy - expected_accuracy) / (1.0 - expected_accuracy)

    row = DataFrame(
        Epoch = epoch,
        Loss = total_loss / length(train_loader),
        Accuracy = accuracy,
        Kappa = kappa,
        F1_micro = f1_micro,
        F1_macro = f1_macro,
        F1_weighted = f1_weighted,
        Precision_micro = precision_micro,
        Precision_macro = precision_macro,
        Precision_weighted = precision_weighted,
        Recall_micro = recall_micro,
        Recall_macro = recall_macro,
        Recall_weighted = recall_weighted,
        Iterations_per_sec = iters_per_sec,
        Epoch_time_sec = duration
    )

    CSV.write(METRICS_CSV, row, append=true)

    # Save confusion matrix and per-class metrics
    CSV.write("confusion_epoch_$epoch.csv", DataFrame(conf_matrix, :auto))
    CSV.write("per_class_epoch_$epoch.csv", DataFrame(
       Class = 1:N_CLASSES, 
       Precision = precision_pc, 
       Recall = recall_pc,
       F1_Score = f1_pc,
       Support = class_support
    ))

    @info @sprintf "[Epoch %d] Loss=%.4f, Acc=%.4f, F1_macro=%.4f, Kappa=%.4f" epoch row.Loss[1] row.Accuracy[1] row.F1_macro[1] row.Kappa[1]
end

Flux.testmode!(model)

# Move to CPU before saving
model_cpu = fmap(cpu, model)
BSON.@save "/tempcnn_julia_model.bson" model_cpu

println("Model saved as 'tempcnn_julia_model.bson'")