library(readxl)
library(dplyr)
library(ggplot2)
library(tidyr)
library(ggrepel)

# 1. Leer archivo
datos <- read_excel("C:/Users/HP/Desktop/proyecto conteo topspin - caribe/pruebas de soporte-modelo/BD_prueba_conteo.xlsx")

# 2. Ver nombres de columnas
names(datos)

# 3. Quedarse solo con filas que tienen conteo real
datos_validos <- datos %>%
  filter(!is.na(N_real))

# 4. Crear variables de error
datos_validos <- datos_validos %>%
  mutate(
    error_modelo = N_modelo - N_real,
    error_operario = N_operario - N_real,
    abs_modelo = abs(error_modelo),
    abs_operario = abs(error_operario),
    error_pct_modelo = abs_modelo / N_real * 100,
    error_pct_operario = abs_operario / N_real * 100
  )

# 5. Resumen general del modelo
resumen_modelo <- datos_validos %>%
  summarise(
    n = n(),
    MAE = mean(abs_modelo, na.rm = TRUE),
    RMSE = sqrt(mean(error_modelo^2, na.rm = TRUE)),
    MAPE = mean(error_pct_modelo, na.rm = TRUE),
    sesgo = mean(error_modelo, na.rm = TRUE),
    sd_error = sd(error_modelo, na.rm = TRUE),
    cor_modelo_real = cor(N_modelo, N_real, use = "complete.obs")
  )

print(resumen_modelo)

# 6. Resumen del operario
resumen_operario <- datos_validos %>%
  filter(!is.na(N_operario)) %>%
  summarise(
    n = n(),
    MAE = mean(abs_operario, na.rm = TRUE),
    RMSE = sqrt(mean(error_operario^2, na.rm = TRUE)),
    MAPE = mean(error_pct_operario, na.rm = TRUE),
    sesgo = mean(error_operario, na.rm = TRUE),
    sd_error = sd(error_operario, na.rm = TRUE),
    cor_operario_real = cor(N_operario, N_real, use = "complete.obs")
  )

print(resumen_operario)

# 7. Base pareada
datos_pareados <- datos_validos %>%
  filter(!is.na(N_operario), !is.na(N_modelo), !is.na(N_real)) %>%
  mutate(
    dif_abs = abs_operario - abs_modelo
  )

# 8. Resumen comparativo pareado
resumen_pareado <- datos_pareados %>%
  summarise(
    n = n(),
    MAE_modelo = mean(abs_modelo),
    MAE_operario = mean(abs_operario),
    MAPE_modelo = mean(error_pct_modelo),
    MAPE_operario = mean(error_pct_operario),
    sesgo_modelo = mean(error_modelo),
    sesgo_operario = mean(error_operario)
  )

print(resumen_pareado)

# 9. Prueba de normalidad
shapiro_result <- shapiro.test(datos_pareados$dif_abs)
print(shapiro_result)

# 10. Wilcoxon pareada
wilcox_result <- wilcox.test(
  datos_pareados$abs_operario,
  datos_pareados$abs_modelo,
  paired = TRUE,
  alternative = "greater"
)
print(wilcox_result)

# 11. t pareada
t_result <- t.test(
  datos_pareados$abs_operario,
  datos_pareados$abs_modelo,
  paired = TRUE,
  alternative = "greater"
)
print(t_result)

# 14. Boxplot de errores absolutos
#------------------------------------
library(dplyr)
library(tidyr)
library(ggplot2)
x11()
# 1. Pasar a formato largo
datos_largos <- datos_pareados %>%
  select(cama, abs_modelo, abs_operario) %>%
  pivot_longer(
    cols = c(abs_modelo, abs_operario),
    names_to = "metodo",
    values_to = "error_absoluto"
  ) %>%
  mutate(
    metodo = factor(
      metodo,
      levels = c("abs_modelo", "abs_operario"),
      labels = c("Modelo", "Operario")
    )
  )

# 2. Prueba estadística pareada
# Usa Wilcoxon porque antes vimos que era la más adecuada
prueba <- wilcox.test(
  datos_pareados$abs_operario,
  datos_pareados$abs_modelo,
  paired = TRUE,
  alternative = "two.sided"
)

p_valor <- prueba$p.value
p_valor

# 3. Crear letras según significancia
resumen_letras <- datos_largos %>%
  group_by(metodo) %>%
  summarise(
    media = mean(error_absoluto, na.rm = TRUE),
    ymax = max(error_absoluto, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(media))

if (p_valor < 0.05) {
  resumen_letras$letra <- c("a", "b")
} else {
  resumen_letras$letra <- c("a", "a")
}

# Ajustar altura de las letras
offset <- 0.06 * max(datos_largos$error_absoluto, na.rm = TRUE)
resumen_letras$ypos <- resumen_letras$ymax + offset

# 4. Colores
colores_metodos <- c(
  "Modelo" = "#26619C",    # azul lapislázuli
  "Operario" = "#FFD60A"   # amarillo pollito
)

# 5. Gráfica
ggplot(datos_largos, aes(x = metodo, y = error_absoluto, color = metodo)) +
  
  # Boxplot con caja blanca y borde de color
  geom_boxplot(
    fill = "white",
    width = 0.48,
    linewidth = 1.2,
    outlier.shape = NA
  ) +
  
  # Puntos individuales
  geom_jitter(
    width = 0.08,
    size = 2.2,
    alpha = 0.75,
    color = "black"
  ) +
  
  # Punto de la media
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 16,
    size = 2.6,
    color = "darkgreen"
  ) +
  
  # Letras de significancia
  geom_text(
    data = resumen_letras,
    aes(x = metodo, y = ypos, label = letra),
    inherit.aes = FALSE,
    size = 6,
    fontface = "bold",
    color = "darkred"
  ) +
  
  scale_color_manual(values = colores_metodos) +
  
  labs(
    title = "Comparación de errores absolutos entre métodos",
    x = "MÉTODOS",
    y = "Error absoluto",
    color = "MÉTODOS"
  ) +
  
  theme_gray(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "plain"),
    axis.title.x = element_text(face = "plain"),
    axis.title.y = element_text(face = "plain"),
    legend.title = element_text(face = "plain"),
    legend.text = element_text(face = "plain"),
    panel.grid.major = element_line(color = "grey60"),
    panel.grid.minor = element_line(color = "grey95")
  )
#-------------------------
# 16. Camas con mayor error del modelo
top_errores <- datos_validos %>%
  arrange(desc(abs_modelo)) %>%
  select(cama, var, bloq, N_real, N_modelo, abs_modelo, error_pct_modelo)

print(top_errores)

# 17. Guardar resultados
write.csv(
  datos_validos,
  "C:/Users/HP/Desktop/proyecto conteo topspin - caribe/pruebas de soporte-modelo/resultados_con_errores.csv",
  row.names = FALSE
)

library(readxl)
library(dplyr)
library(ggplot2)
library(ggrepel)

# =========================================================
# 1. MODELO VS REAL
# =========================================================
datos_modelo <- datos %>%
  filter(!is.na(N_real), !is.na(N_modelo))

errores_abs_modelo <- abs(datos_modelo$N_modelo - datos_modelo$N_real)

# Equivalente exacto a Excel: DESVEST.P
sd_error_modelo <- sqrt(
  sum((errores_abs_modelo - mean(errores_abs_modelo, na.rm = TRUE))^2, na.rm = TRUE) /
    sum(!is.na(errores_abs_modelo))
)

x11()
ggplot(datos_modelo, aes(x = N_real, y = N_modelo)) +
  geom_segment(
    aes(x = N_real, xend = N_real, y = N_real, yend = N_modelo),
    color = "gray50",
    linewidth = 0.7
  ) +
  geom_abline(
    intercept = sd_error_modelo,
    slope = 1,
    linetype = "dashed",
    color = "blue",
    linewidth = 0.7
  ) +
  geom_abline(
    intercept = -sd_error_modelo,
    slope = 1,
    linetype = "dashed",
    color = "blue",
    linewidth = 0.7
  ) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    color = "red",
    linewidth = 0.9
  ) +
  geom_point(size = 3, color = "black") +
  geom_text_repel(aes(label = cama), size = 4, max.overlaps = 100) +
  labs(
    title = "Conteo del modelo vs conteo real",
    subtitle = paste0(
      "Línea roja: igualdad perfecta | Líneas azules: ±1 DE del error (",
      round(sd_error_modelo, 2), ")"
    ),
    x = "Conteo real",
    y = "Conteo modelo"
  ) +
  theme_minimal(base_size = 14)

# =========================================================
# 2. OPERARIO VS REAL
# =========================================================
datos_operario <- datos %>%
  filter(!is.na(N_real), !is.na(N_operario))

errores_abs_operario <- abs(datos_operario$N_operario - datos_operario$N_real)

# Equivalente exacto a Excel: DESVEST.P
sd_error_operario <- sqrt(
  sum((errores_abs_operario - mean(errores_abs_operario, na.rm = TRUE))^2, na.rm = TRUE) /
    sum(!is.na(errores_abs_operario))
)

x11()
ggplot(datos_operario, aes(x = N_real, y = N_operario)) +
  geom_segment(
    aes(x = N_real, xend = N_real, y = N_real, yend = N_operario),
    color = "gray50",
    linewidth = 0.7
  ) +
  geom_abline(
    intercept = sd_error_operario,
    slope = 1,
    linetype = "dashed",
    color = "blue",
    linewidth = 0.7
  ) +
  geom_abline(
    intercept = -sd_error_operario,
    slope = 1,
    linetype = "dashed",
    color = "blue",
    linewidth = 0.7
  ) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    color = "red",
    linewidth = 0.9
  ) +
  geom_point(size = 3, color = "black") +
  geom_text_repel(aes(label = cama), size = 4, max.overlaps = 100) +
  labs(
    title = "Conteo del operario vs conteo real",
    subtitle = paste0(
      "Línea roja: igualdad perfecta | Líneas azules: ±1 DE del error (",
      round(sd_error_operario, 2), ")"
    ),
    x = "Conteo real",
    y = "Conteo operario"
  ) +
  theme_minimal(base_size = 14)

