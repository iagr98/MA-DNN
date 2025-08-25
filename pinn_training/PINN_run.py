from pinn_training.PINN_model import PINN

N_colloc = 21
filename = 'in_silico_dataset.csv'
hidden_layers = [128, 128, 128]

pinn = PINN(N_colloc, filename, hidden_layers)
pinn.create_model()
pinn.pre_training(epochs=10, lr=1e-3, data_loss_batch=32)
pinn.training(epochs=2, lr=5e-4, ode_loss_batch=10, bc_loss_batch=32)