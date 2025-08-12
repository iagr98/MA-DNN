from PINN_model import PINN

N_colloc = 21
filename = 'in_silico_dataset.csv'
data_loss_batch = 32
hidden_layers = [128, 128, 128]

pinn = PINN(N_colloc, filename, data_loss_batch, hidden_layers)
pinn.create_model()
pinn.pre_training(epochs=10, lr=1e-3)
pinn.training(epochs=1, lr=1e-4)