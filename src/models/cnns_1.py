# Comparison of different CNN architectures for the single-fermion system.
# Model 1 consists of a single convolutional layer followed by max pooling and a fully connected neural network.
# Subsequent models progressively increase the number of convolutional and max-pooling blocks.

from typing import Any
import torch.nn as nn

hidden_dim = 60 #Hidden layer dimension

class modello1(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.sequenziale = nn.Sequential( #input shape (batch, canale, 18,18,18)
            nn.Conv3d(in_channels=1, out_channels=hidden_dim, kernel_size=5, stride=1, padding_mode='circular'),
           #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=5),
            # nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim,kernel_size=4, stride=1, padding_mode='circular'),
            # #nn.BatchNorm3d(hidden_dim),
            # nn.Softplus(),
            # nn.MaxPool3d(kernel_size=2), #2x2x2
            nn.Flatten(), #480

            nn.Linear(480, 240),
            nn.Softplus(),
            nn.Linear(240,120),
            nn.Softplus(),
            nn.Linear(120, 60),
            nn.Softplus(),
            nn.Linear(60, 30),
            nn.Softplus(),
            nn.Linear(30, 10),
            nn.Softplus(),
            nn.Linear(10, 1),
        )


    def forward(self, x):
        x = self.sequenziale(x)
        #print(x.shape)
        return x

class modello2(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.sequenziale = nn.Sequential( #input shape (batch, canale, 18,18,18)
            nn.Conv3d(in_channels=1, out_channels=hidden_dim, kernel_size=5, stride=1, padding_mode='circular'),
           #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim,kernel_size=4, stride=1, padding_mode='circular'),
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2), #2x2x2
            nn.Flatten(), #480

            nn.Linear(480, 240),
            nn.Softplus(),
            nn.Linear(240,120),
            nn.Softplus(),
            nn.Linear(120, 60),
            nn.Softplus(),
            nn.Linear(60, 30),
            nn.Softplus(),
            nn.Linear(30, 10),
            nn.Softplus(),
            nn.Linear(10, 1),
        )


    def forward(self, x):
        x = self.sequenziale(x)
        #print(x.shape)
        return x



class modello3(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.sequenziale = nn.Sequential( #input shape (batch, canale, 18,18,18)
            nn.Conv3d(in_channels=1, out_channels=hidden_dim, kernel_size=2, stride=1, padding=1,padding_mode='circular'),
           #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim,kernel_size=2, stride=1,padding=1, padding_mode='circular'),
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2), #2x2x2
            nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim,kernel_size=2, stride=1, padding_mode='circular'),
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2), #2x2x2
            nn.Flatten(), #60

            nn.Linear(480, 240),
            nn.Softplus(),
            nn.Linear(240,120),
            nn.Softplus(),
            nn.Linear(120, 60),
            nn.Softplus(),
            nn.Linear(60, 30),
            nn.Softplus(),
            nn.Linear(30, 10),
            nn.Softplus(),
            nn.Linear(10, 1),
        )


    def forward(self, x):
        x = self.sequenziale(x)
        #print(x.shape)
        return x



class modello4(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.sequenziale = nn.Sequential( #input shape (batch, canale, 18,18,18)
            nn.Conv3d(
                1,
                hidden_dim, 
                kernel_size=3, 
                stride=1, 
                padding=0, 
                padding_mode="circular"
        ),  # 18x18x18 -> 18x18x18
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=1), #18x18x18 -> 10x10x10
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=0,
                padding_mode="circular",
            ),  # 10x10x10 -> 10x10x10
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1), #10x10x10 -> 6x6x6
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=0,
                padding_mode="circular",
            ),  # 6x6x6 -> 6x6x6
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=0), #6x6x6 -> 4x4x4
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),  # 4x4x4 -> 4x4x4
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=0),  #4x4x4 -> 3x3x3
            nn.Flatten(), #480

            nn.Linear(480, 240),
            nn.Softplus(),
            nn.Linear(240,120),
            nn.Softplus(),
            nn.Linear(120, 60),
            nn.Softplus(),
            nn.Linear(60, 30),
            nn.Softplus(),
            nn.Linear(30, 10),
            nn.Softplus(),
            nn.Linear(10, 1),
        )


    def forward(self, x):
        x = self.sequenziale(x)
        #print(x.shape)
        return x


#2 Livelli con e maxpool, 6 deep


class modello5(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.sequenziale = nn.Sequential( #input shape (batch, canale, 18,18,18)
            nn.Conv3d(
                1,
                hidden_dim, 
                kernel_size=3, 
                stride=1, 
                padding=0, 
                padding_mode="circular"
        ),  # 18x18x18 -> 18x18x18
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=1), #18x18x18 -> 10x10x10
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=0,
                padding_mode="circular",
            ),  # 10x10x10 -> 10x10x10
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1), #10x10x10 -> 6x6x6
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="circular",
            ),  # 6x6x6 -> 6x6x6
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=1), #6x6x6 -> 4x4x4
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),  # 4x4x4 -> 4x4x4
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=0),  #4x4x4 -> 3x3x3
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),  # 4x4x4 -> 4x4x4
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=0),  #4x4x4 -> 3x3x3
            
            
            nn.Flatten(), #480

            nn.Linear(480, 240),
            nn.Softplus(),
            nn.Linear(240,120),
            nn.Softplus(),
            nn.Linear(120, 60),
            nn.Softplus(),
            nn.Linear(60, 30),
            nn.Softplus(),
            nn.Linear(30, 10),
            nn.Softplus(),
            nn.Linear(10, 1),
        )


    def forward(self, x):
        x = self.sequenziale(x)
        #print(x.shape)
        return x




class modello6(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.sequenziale = nn.Sequential( #input shape (batch, canale, 18,18,18)
            nn.Conv3d(
                1,
                hidden_dim, 
                kernel_size=3, 
                stride=1, 
                padding=0, 
                padding_mode="circular"
        ),  # 18x18x18 -> 18x18x18
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=1), #18x18x18 -> 10x10x10
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=0,
                padding_mode="circular",
            ),  # 10x10x10 -> 10x10x10
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1), #10x10x10 -> 6x6x6
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="circular",
            ),  # 6x6x6 -> 6x6x6
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=1), #6x6x6 -> 4x4x4
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),  # 4x4x4 -> 4x4x4
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=0),  #4x4x4 -> 3x3x3
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),  # 4x4x4 -> 4x4x4
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=3, stride=1, padding=0),  #4x4x4 -> 3x3x3
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),  # 4x4x4 -> 4x4x4
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=1),  #4x4x4 -> 3x3x3
            
            
            nn.Flatten(), #480

            nn.Linear(480, 240),
            nn.Softplus(),
            nn.Linear(240,120),
            nn.Softplus(),
            nn.Linear(120, 60),
            nn.Softplus(),
            nn.Linear(60, 30),
            nn.Softplus(),
            nn.Linear(30, 10),
            nn.Softplus(),
            nn.Linear(10, 1),
        )


    def forward(self, x):
        x = self.sequenziale(x)
        #print(x.shape)
        return x



class modello7(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.sequenziale = nn.Sequential( #input shape (batch, canale, 18,18,18)
            nn.Conv3d(
                1,
                hidden_dim, 
                kernel_size=3, 
                stride=1, 
                padding=0, 
                padding_mode="circular"
        ),  # 18x18x18 -> 18x18x18
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=1), #18x18x18 -> 10x10x10
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=0,
                padding_mode="circular",
            ),  # 10x10x10 -> 10x10x10
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=1), #10x10x10 -> 6x6x6
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="circular",
            ),  # 6x6x6 -> 6x6x6
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=0), #6x6x6 -> 4x4x4
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),  # 4x4x4 -> 4x4x4
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=0),  #4x4x4 -> 3x3x3
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),  # 4x4x4 -> 4x4x4
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=1),  #4x4x4 -> 3x3x3
            
            
            nn.Flatten(), 

            nn.Linear(1620, 1000),
            nn.Softplus(),
            nn.Linear(1000,500),
            nn.Softplus(),
            nn.Linear(500, 250),
            nn.Softplus(),
            nn.Linear(250,100),
            nn.Softplus(),
            nn.Linear(100, 25),
            nn.Softplus(),
            nn.Linear(25, 1),
        )


    def forward(self, x):
        x = self.sequenziale(x)
        #print(x.shape)
        return x


class modello8(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.sequenziale = nn.Sequential( #input shape (batch, canale, 18,18,18)
            nn.Conv3d(
                1,
                hidden_dim, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                padding_mode="circular"
        ),  # 18x18x18 -> 18x18x18
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=1), #18x18x18 -> 10x10x10
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),  # 10x10x10 -> 10x10x10
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=1), #10x10x10 -> 6x6x6
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="circular",
            ),  # 6x6x6 -> 6x6x6
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=0), #6x6x6 -> 4x4x4
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),  # 4x4x4 -> 4x4x4
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            nn.MaxPool3d(kernel_size=2, stride=1, padding=0),  #4x4x4 -> 3x3x3
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),  # 4x4x4 -> 4x4x4
            #nn.BatchNorm3d(hidden_dim),
            nn.Softplus(),
            #nn.MaxPool3d(kernel_size=2, stride=1, padding=0),  #4x4x4 -> 3x3x3
            nn.AdaptiveMaxPool3d(1),
            
            nn.Flatten(), 

            nn.Linear(60, 50),
            nn.Softplus(),
            nn.Linear(50,40),
            nn.Softplus(),
            nn.Linear(40, 30),
            nn.Softplus(),
            nn.Linear(30,20),
            nn.Softplus(),
            nn.Linear(20, 10),
            nn.Softplus(),
            nn.Linear(10, 1),
        )


    def forward(self, x):
        x = self.sequenziale(x)
        #print(x.shape)
        return x

