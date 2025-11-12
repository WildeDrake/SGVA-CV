import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist


### Este modelo es implementado basándose en el artículo: "PROJECTING SUPERFICIAL STATISTICS OUT"
'''
Módulo | Rol	                | Tipo	      | Entrada 	           | Salida
qzd	   | Encoder de sujeto	    | Inferencial | x	                   | zd
qzx	   | Encoder de muestra	    | Inferencial | x	                   | zx
qzy	   | Encoder de gesto	    | Inferencial | x	                   | zy
px	   | Decoder	            | Generativo  | zd, zx, zy	           | x_recon
pzd	   | Generador latente	    | Generativo  | d (one-hot del sujeto) | zd
pzy	   | Generador latente	    | Generativo  | y (one-hot del gesto)  | zy
qd	   | Clasificador auxiliar	| Auxiliar	  | zd	                   | d_hat
qy	   | Clasificador auxiliar	| Auxiliar	  | zy	                   | y_hat
'''


#------------------------------ Encoders  ------------------------------#

# Encoder de sujeto
class qzd(nn.Module):
    '''
    params:
        zd_dim: tamaño del espacio latente de la etiqueta de sujeto
    return:
        zd_loc: Media de la distribución del espacio latente
        zd_scale: Desviación estándar de la distribución del espacio latente
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzd, self).__init__()
        # 1024*1*8*52
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=(5, 13), stride=1, bias=False), nn.BatchNorm2d(12), nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # 1024*12*4*20
            nn.Conv2d(12, 24, kernel_size=(4, 11), stride=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(),
            # 1024*24*1*10
        )
        self.fc11 = nn.Sequential(nn.Linear(240, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(240, zd_dim), nn.Softplus())
        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.encoder(x)  # 1024*24*1*10
        h = h.view(-1, 24 * 1 * 10)
        zd_loc = self.fc11(h)
        zd_scale = self.fc12(h) + 1e-7
        return zd_loc, zd_scale

# Encoder de muestra
class qzx(nn.Module):
    '''
    params:
        zx_dim: tamaño del espacio latente de la etiqueta de muestra
    return:
        zx_loc: Media de la distribución del espacio latente
        zx_scale: Desviación estándar de la distribución del espacio latente
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzx, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=(5, 13), stride=1, bias=False), nn.BatchNorm2d(12), nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # 1024*12*4*20
            nn.Conv2d(12, 24, kernel_size=(4, 11), stride=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(),
            # 1024*24*1*10
        )
        self.fc11 = nn.Sequential(nn.Linear(240, zx_dim))
        self.fc12 = nn.Sequential(nn.Linear(240, zx_dim), nn.Softplus())
        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)  # Inicializar los pesos de las dos capas convolucionales
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.encoder(x)  # 1024*24*1*10
        h = h.view(-1, 24 * 1 * 10)
        zx_loc = self.fc11(h)
        zx_scale = self.fc12(h) + 1e-7
        return zx_loc, zx_scale

# Encoder de gesto
class qzy(nn.Module):
    '''
    params:
        zy_dim: tamaño del espacio latente de la etiqueta de gesto
    return:
        zy_loc: Media de la distribución del espacio latente
        zy_scale: Desviación estándar de la distribución del espacio latente
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzy, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=(5, 13), stride=1, bias=False), nn.BatchNorm2d(12), nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # 1024*12*4*20
            nn.Conv2d(12, 24, kernel_size=(4, 11), stride=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(),
            # 1024*24*1*10
        )
        self.fc11 = nn.Sequential(nn.Linear(240, zy_dim))
        self.fc12 = nn.Sequential(nn.Linear(240, zy_dim), nn.Softplus())
        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, 240)
        zy_loc = self.fc11(h)
        zy_scale = self.fc12(h) + 1e-7
        return zy_loc, zy_scale


#------------------------------- Decoder  ------------------------------#

# Decoder Principal
class px(nn.Module):
    '''
    params:
        zd_dim: tamaño del espacio latente de la etiqueta de sujeto
        zx_dim: tamaño del espacio latente de la muestra
        zy_dim: tamaño del espacio latente de la etiqueta de gesto
    return:
        Reconstrucción de la señal de entrada
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(px, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(zd_dim + zx_dim + zy_dim, 240, bias=False), nn.BatchNorm1d(240), nn.ReLU())
        self.de1 = nn.Sequential(nn.ConvTranspose2d(24, 36, kernel_size=(4, 11), stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(36), nn.ReLU())
        self.up2 = nn.Upsample([4, 40])
        self.de2 = nn.Sequential(nn.ConvTranspose2d(36, 48, kernel_size=(5, 13), stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(48), nn.ReLU())
        self.de3 = nn.Sequential(nn.Conv2d(48, 48, kernel_size=1, stride=1))
        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.de1[0].weight)
        torch.nn.init.xavier_uniform_(self.de2[0].weight)
        torch.nn.init.xavier_uniform_(self.de3[0].weight)
        self.de3[0].bias.data.zero_()

    def forward(self, zd, zx, zy):
        if zx is None:
            zdzxzy = torch.cat((zd, zy), dim=-1)
        else:
            zdzxzy = torch.cat((zd, zx, zy), dim=-1)
        h = self.fc1(zdzxzy)  # 1024*240
        h = h.view(-1, 24, 1, 10)  # 1024*24*1*10
        h = self.de1(h)  # 1024*36*4*20
        h = self.up2(h)
        h = self.de2(h)
        loc_img = self.de3(h)
        return loc_img  # 1024*48*8*52


#--------------------------- Generador latente ------------------------------#

# decoder de etiquetas de dominio de sujeto
class pzd(nn.Module):
    '''
    Mapeo de la etiqueta de sujeto a una distribución en el espacio latente zd
        d_dim: número de dominios de origen
        zd_dim: tamaño del espacio latente de la etiqueta de sujeto
    return:
        zd_loc: Media de la distribución del espacio latente
        zd_scale: Desviación estándar de la distribución del espacio latente
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(pzd, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(d_dim, zd_dim, bias=False), nn.BatchNorm1d(zd_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zd_dim, zd_dim))
        self.fc22 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())
        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, d):  # 1024*6
        hidden = self.fc1(d)
        zd_loc = self.fc21(hidden)
        zd_scale = self.fc22(hidden) + 1e-7
        return zd_loc, zd_scale

# Decoder de etiquetas de gesto
class pzy(nn.Module):
    '''
    Mapeo de la etiqueta de gesto a una distribución en el espacio latente zy
    params:
        y_dim: número de clases de gesto
        zy_dim: tamaño del espacio latente de la etiqueta de gesto
    return:
        zy_loc: Media de la distribución del espacio latente
        zy_scale: Desviación estándar de la distribución del espacio latente
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(pzy, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(y_dim, zy_dim, bias=False), nn.BatchNorm1d(zy_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zy_dim, zy_dim))
        self.fc22 = nn.Sequential(nn.Linear(zy_dim, zy_dim), nn.Softplus())
        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, y):
        hidden = self.fc1(y)
        zy_loc = self.fc21(hidden)
        zy_scale = self.fc22(hidden) + 1e-7

        return zy_loc, zy_scale
    

#--------------------------- Classificadores Auxiliares -------------------------#

# Classificador de sujeto
class qd(nn.Module):
    '''
    params:
        d_dim: tamaño del espacio de la etiqueta de sujeto
        zd_dim: tamaño del espacio latente de la etiqueta de sujeto
    return:
        loc_d: etiqueta de sujeto predicha
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qd, self).__init__()
        self.fc1 = nn.Linear(zd_dim, d_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zd):
        h = F.relu(zd)
        loc_d = self.fc1(h)
        return loc_d

# Classificador de gesto
class qy(nn.Module):
    '''
    params:
        y_dim: tamaño del espacio de la etiqueta de gesto
        zy_dim: tamaño del espacio latente de la etiqueta de gesto
    return:
        loc_y: etiqueta de gesto predicha
    '''
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qy, self).__init__()
        self.fc1 = nn.Linear(zy_dim, y_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        # torch.nn.init.xavier_uniform_(self.dense_layer.weight)
        # self.dense_layer.bias.data.zero_()

    def forward(self, zy):
        h = F.relu(zy)
        loc_y = self.fc1(h)
        return loc_y


#--------------------------- Modelo DIVA Completo ------------------------------#

class DIVA(nn.Module):
    # Constructor
    def __init__(self, args):
        super(DIVA, self).__init__()
        # Parámetros del modelo
        self.zd_dim = args.zd_dim  # tamaño del espacio latente de la etiqueta de sujeto
        self.zx_dim = args.zx_dim  # tamaño del espacio latente de la etiqueta de contexto
        self.zy_dim = args.zy_dim  # tamaño del espacio latente de la etiqueta de gesto
        self.d_dim = args.d_dim  # número de dominios
        self.x_dim = args.x_dim  # tamaño de entrada después de aplanar
        self.y_dim = args.y_dim  # número de clases
        # Índices de inicio de cada variable latente en el vector concatenado
        self.start_zx = self.zd_dim
        self.start_zy = self.zd_dim + self.zx_dim
        # Construir los módulos del modelo
        self.px = px(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.pzd = pzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.pzy = pzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        # Classificadores
        self.qzd = qzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        if self.zx_dim != 0:
            self.qzx = qzx(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qzy = qzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        # Classificador de sujeto
        self.qd = qd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qy = qy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        # Multiplicadores de pérdida auxiliar
        self.aux_loss_multiplier_y = args.aux_loss_multiplier_y
        self.aux_loss_multiplier_d = args.aux_loss_multiplier_d
        # Pesos de las pérdidas KL
        self.beta_d = args.beta_d
        self.beta_x = args.beta_x
        self.beta_y = args.beta_y
        # Mover a GPU
        self.cuda()

    # Forward
    def forward(self, d, x, y):
        """
        x_recon, d_hat, y_hat # reconstrucción y etiquetas predichas
        zd_q, zx_q, zy_q # muestras latentes del modelo de inferencia
        """
        # x -> zd zx zy ->x_recon(inference model -> generative model)
        # 1 Encode : obtener parámetros de la distribución gaussiana
        zd_q_loc, zd_q_scale = self.qzd(x)  # 1024*1*8*52 -> 1024*64
        if self.zx_dim != 0:
            zx_q_loc, zx_q_scale = self.qzx(x)  # 1024*64
        zy_q_loc, zy_q_scale = self.qzy(x)  # 1024*64

        # 2 Reparameterization trick: Creando distribución Gaussiana
        qzd = dist.Normal(zd_q_loc, zd_q_scale)  # 1024*64
        zd_q = qzd.rsample()
        if self.zx_dim != 0:
            qzx = dist.Normal(zx_q_loc, zx_q_scale)  # 1024*64
            zx_q = qzx.rsample()
        else:
            qzx = None
            zx_q = None
        qzy = dist.Normal(zy_q_loc, zy_q_scale)  # 1024*64
        zy_q = qzy.rsample()

        # 3 Decode:  generar la reconstrucción
        x_recon = self.px(zd_q, zx_q, zy_q)  # 1024*48*8*52
        # d,y ->zd,zy -> d_hat,y_hat    (generative model ->inference model)
        zd_p_loc, zd_p_scale = self.pzd(d)
        if self.zx_dim != 0:
            zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim).cuda(), \
                torch.ones(zd_p_loc.size()[0], self.zx_dim).cuda()
        zy_p_loc, zy_p_scale = self.pzy(y)

        # 4 Reparameterization trick: Creando distribución Gaussiana
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        if self.zx_dim != 0:
            pzx = dist.Normal(zx_p_loc, zx_p_scale)
        else:
            pzx = None
        pzy = dist.Normal(zy_p_loc, zy_p_scale)

        # -------- Losses auxiliares ---------#
        d_hat = self.qd(zd_q)
        y_hat = self.qy(zy_q)
        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q

    # Pérdida
    def loss_function(self, d, x, y=None):
        if y is None:  # unsupervised
            # reconstruction loss
            zd_q_loc, zd_q_scale = self.qzd(x)
            if self.zx_dim != 0:
                zx_q_loc, zx_q_scale = self.qzx(x)
            zy_q_loc, zy_q_scale = self.qzy(x)
            # Reparameterization trick
            qzd = dist.Normal(zd_q_loc, zd_q_scale)
            zd_q = qzd.rsample()
            if self.zx_dim != 0:
                qzx = dist.Normal(zx_q_loc, zx_q_scale)
                zx_q = qzx.rsample()
            else:
                zx_q = None
            qzy = dist.Normal(zy_q_loc, zy_q_scale)
            zy_q = qzy.rsample()
            # Decode
            zd_p_loc, zd_p_scale = self.pzd(d)
            if self.zx_dim != 0:
                zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim).cuda(), \
                    torch.ones(zd_p_loc.size()[0], self.zx_dim).cuda()
            pzd = dist.Normal(zd_p_loc, zd_p_scale)
            # Reparameterization trick
            if self.zx_dim != 0:
                pzx = dist.Normal(zx_p_loc, zx_p_scale)
            else:
                pzx = None
            d_hat = self.qd(zd_q)
            # Reconstruction
            x_recon = self.px(zd_q, zx_q, zy_q)
            x_recon = x_recon.view(-1, 256)
            x_target = (x.view(-1) * 255).long()
            CE_x = F.cross_entropy(x_recon, x_target, reduction='sum')
            # KL Divergence
            zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
            if self.zx_dim != 0:
                KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
            else:
                KL_zx = 0
            # classification loss of subject
            _, d_target = d.max(dim=1)
            CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')
            # Crear onehot para todas las clases de gesto
            y_onehot = torch.eye(10)
            y_onehot = y_onehot.repeat(1, 100)
            y_onehot = y_onehot.view(1000, 10).cuda()
            # Repetir zy_q para cada sujeto
            zy_q = zy_q.repeat(10, 1)
            zy_q_loc, zy_q_scale = zy_q_loc.repeat(10, 1), zy_q_scale.repeat(10, 1)
            qzy = dist.Normal(zy_q_loc, zy_q_scale)
            # Do forward pass for everything involving y
            zy_p_loc, zy_p_scale = self.pzy(y_onehot)
            # Reparameterization trick
            pzy = dist.Normal(zy_p_loc, zy_p_scale)
            # Auxiliary losses
            y_hat = self.qy(zy_q)
            # Marginals
            alpha_y = F.softmax(y_hat, dim=-1)
            qy = dist.OneHotCategorical(alpha_y)
            prob_qy = torch.exp(qy.log_prob(y_onehot))
            # KL Divergence
            zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q), dim=-1)
            marginal_zy_p_minus_zy_q = torch.sum(prob_qy * zy_p_minus_zy_q)
            prior_y = torch.tensor(1 / 10).cuda()
            prior_y_minus_qy = torch.log(prior_y) - qy.log_prob(y_onehot)
            marginal_prior_y_minus_qy = torch.sum(prob_qy * prior_y_minus_qy)
            # Total loss
            return CE_x \
                - self.beta_d * zd_p_minus_zd_q \
                - self.beta_x * KL_zx \
                - self.beta_y * marginal_zy_p_minus_zy_q \
                - marginal_prior_y_minus_qy \
                + self.aux_loss_multiplier_d * CE_d

        else:  # supervised
            # reconstruction loss
            x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = self.forward(d, x, y)
            x_recon = x_recon.permute(0, 2, 3, 1)
            x_recon = x_recon.reshape(-1, 48)
            # x_recon = x_recon.view(-1, 48)
            x_target = (x.permute(0, 1, 3, 2).reshape(-1)).long()
            CE_x = F.cross_entropy(x_recon, x_target, reduction='sum')
            # KL Divergence
            zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
            if self.zx_dim != 0:
                KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))  # -KL
            else:
                KL_zx = 0
            zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))
            # classification loss of subject and gesture
            _, d_target = d.max(dim=1)
            CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')
            _, y_target = y.max(dim=1)
            CE_y = F.cross_entropy(y_hat, y_target, reduction='sum')
            # Total loss
            return CE_x \
                   - self.beta_d * zd_p_minus_zd_q \
                   - self.beta_x * KL_zx \
                   - self.beta_y * zy_p_minus_zy_q \
                   + self.aux_loss_multiplier_d * CE_d \
                   + self.aux_loss_multiplier_y * CE_y, \
                CE_y, zd_q, zy_q, zx_q, d_target, y_target

    # Clasificador
    def classifier(self, x):
        """
        clasificar una imagen (o un lote de imágenes)
        :param xs: un lote de vectores escalados de píxeles de una imagen
        :return: un lote de las correspondientes etiquetas de clase (como one-hots)
        """
        with torch.no_grad():
            zd_q_loc, zd_q_scale = self.qzd(x)
            zd = zd_q_loc
            alpha = F.softmax(self.qd(zd), dim=1)
            # obtiener el índice (dígito) que corresponde a la máxima probabilidad de clase predicha
            res, ind = torch.topk(alpha, 1)
            # Convirtir el(los) dígito(s) a tensor(es) one-hot
            d = x.new_zeros(alpha.size())
            d = d.scatter_(1, ind, 1.0)
            # Obtener la etiqueta de gesto
            zy_q_loc, zy_q_scale = self.qzy.forward(x)
            zy = zy_q_loc
            alpha = F.softmax(self.qy(zy), dim=1)
            # obtener el índice (dígito) que corresponde a la máxima probabilidad de clase predicha
            res, ind = torch.topk(alpha, 1)
            # convertir el(los) dígito(s) a tensor(es) one-hot
            y = x.new_zeros(alpha.size())
            y = y.scatter_(1, ind, 1.0)
        # retornar las etiquetas de clase
        return d, y

    # Clasificador para t-SNE
    def classifier_for_tsne(self, x):
        """
        clasificar una imagen (o un lote de imágenes)
        :param xs: un lote de vectores escalados de píxeles de una imagen
        :return: un lote de las correspondientes etiquetas de clase (como one-hots)
        """
        with torch.no_grad():
            zd_q_loc, zd_q_scale = self.qzd(x)
            zd = zd_q_loc
            alpha = F.softmax(self.qd(zd), dim=1)
            # obtener el índice (dígito) que corresponde a la máxima probabilidad de clase predicha
            res, ind = torch.topk(alpha, 1) 
            # convertir el(los) dígito(s) a tensor(es) one-hot
            d = x.new_zeros(alpha.size())
            d = d.scatter_(1, ind, 1.0)
            # Obtener la etiqueta de gesto
            zy_q_loc, zy_q_scale = self.qzy.forward(x)
            zy = zy_q_loc
            alpha = F.softmax(self.qy(zy), dim=1)
            # obtener el índice (dígito) que corresponde a la máxima probabilidad de clase predicha
            res, ind = torch.topk(alpha, 1)
            # convertir el(los) dígito(s) a tensor(es) one-hot
            y = x.new_zeros(alpha.size())
            y = y.scatter_(1, ind, 1.0)
        # retornar las etiquetas de clase
        return d, y, zy
