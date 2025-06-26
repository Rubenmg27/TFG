import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class MyLightningModule(pl.LightningModule):
    def __init__(self, model, loss_fn, train_dataset, val_dataset, learning_rate=0.001):
        super(MyLightningModule, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_outputs = []  # Atributo para almacenar los outputs de validación

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self.forward(X_batch)
        loss = self.loss_fn(outputs, y_batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self.forward(X_batch)
        val_loss = self.loss_fn(outputs, y_batch)

        # Guardar las predicciones para el cálculo final en `on_validation_epoch_end`
        predicted = torch.round(outputs)
        correct = (predicted == y_batch).sum().item()
        total = y_batch.size(0)
        val_accuracy = correct / total

        self.val_outputs.append({"val_loss": val_loss, "val_accuracy": val_accuracy})  # Almacenar resultados

    def on_validation_epoch_end(self):
        # Al final de cada época de validación, calculamos el promedio de las pérdidas y precisión
        avg_val_loss = torch.stack([x["val_loss"] for x in self.val_outputs]).mean()
        avg_val_accuracy = sum([x["val_accuracy"] for x in self.val_outputs]) / len(self.val_outputs)

        self.log("val_loss", avg_val_loss)
        self.log("val_accuracy", avg_val_accuracy)

        # Limpiar los resultados almacenados al final de la época
        self.val_outputs.clear()

    def configure_optimizers(self):
        # Definimos el optimizador
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        # Retornamos el DataLoader para el conjunto de entrenamiento
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True)

    def val_dataloader(self):
        # Retornamos el DataLoader para el conjunto de validación
        return DataLoader(self.val_dataset, batch_size=1)
