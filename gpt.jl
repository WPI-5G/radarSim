using Flux

# Define the RNN model
m = Chain(
    LSTM(1, 32),
    LSTM(32, 32),
    Dense(32, 1),
)

# Define the loss function
loss(x, y) = Flux.mse(m(x), y)

# Define the optimizer
opt =  Flux.setup(Adam(), m)

# Generate a random time series of length 5000
x = rand(5000)

# Define the target output as the maximum value in the time series
y = maximum(x)

# Train the RNN model using the time series and target output
# for i in 1:1000
#     Flux.train!(loss, [(x, y)], opt)
# end

# Use the trained model to predict the maximum value in the time series
m(x)
