from score import init, run
# (Re-)initialize any global state
init()  
# Create a dummy MNIST-style payload:
sample_payload = {"data": [[0.0]*784]}  
result = run(sample_payload)  
print("Inference result:", result)
