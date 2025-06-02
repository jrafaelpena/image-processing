# Taller Registro
**Asignatura:** Procesamiento de imágenes médicas.

**Autores:** José Rafael Peña Gutiérrez, Luz Andrea García.

**Fecha:** 01/06/2025

## Paso 0:

Cabe resaltar que inicialmente se realizó el ejericio en Slicer3D y se exploraron los parámetros y componentes default usados en el programa para poder tomar la decisión de cuáles usar directamente en ITK. A continuación se identificaron los siguientes parámetros:

<img src="params_1.png" width="70%">
<img src="params_2.png" width="75%">

## Paso 1: Rigid

En el primer paso aplicamos una transformación rígida, es decir, solo permitimos rotación y traslación del volumen en el espacio. Este registro se emplea comúnmente como etapa inicial para alinear de forma general dos imágenes volumétricas.

Los componentes usados en ITK son los siguientes:

- **Transform:** `VersorRigid3DTransform`, preferido por su compatibilidad con `CenteredTransformInitializer`. Se eligió sobre `Euler3DTransform` ya que este último no funcionaba adecuadamente con el inicializador centrado.
- **Optimizer:** `RegularStepGradientDescentOptimizerv4`, que permite realizar pasos controlados y estables. Fue también utilizado en el ejercicio en clase.
- **Metric:** `MattesMutualInformationImageToImageMetricv4`, que mide la similitud entre imágenes usando información mutua, ideal para imágenes de diferentes sujetos.
- **Registration:** `ImageRegistrationMethodv4`, el contenedor general que une todos los componentes.

### Parámetros usados

**Metric**

- `SetNumberOfHistogramBins(50)`: Valor extraído directamente de Slicer3D.
- `SetMetricSamplingPercentage`: Se calcula a partir de la razón entre `number_of_samples` y el tamaño total de la imagen fija. Se probó inicialmente con 400,000 muestras, pero se redujo a 100,000 por motivos de tiempo computacional.

**Optimizer**

- `SetLearningRate(0.2)`: Extraído del ejemplo usado en clase.
- `SetMinimumStepLength(0.01)`: Se incrementa respecto a los parámetros de Slicer3D para menor tiempo.
- `SetNumberOfIterations(1500)`: Extraído de Slicer3D.
- `SetRelaxationFactor(0.5)`: Valor por defecto en muchos ejercicios prácticos.

Además, se utilizó `CenteredTransformInitializer` para inicializar la transformación basada en los momentos de intensidad de ambas imágenes (fija y móvil), lo cual proporciona una alineación inicial razonable antes de optimizar.

### Iteraciones
<img src="rigid_iterations.png" width="90%">

## Paso 2: Similarity (Rigid + Scale)

En el segundo paso del proceso de registro, ampliamos el modelo rígido para incluir además un factor de escala. Este paso permite compensar diferencias globales de tamaño entre las imágenes, por ejemplo, causadas por variaciones entre sujetos o por diferencias en la adquisición.

### Transformación Similarity

Usamos la clase `Similarity3DTransform`, que extiende la transformación rígida añadiendo un solo parámetro de escala uniforme. Esto permite mantener la estructura general de la imagen (sin cizallamiento) pero ajustarla en tamaño.

La transformación inicial se construye reutilizando la salida del paso rígido anterior. Para ello:

- Se extrae el centro, la rotación (como versor) y la traslación de la transformación rígida anterior.
- Se inicializa la escala en 1.0 (sin escalado).
- Estos valores se asignan directamente al nuevo objeto `Similarity3DTransform`.

Este diseño garantiza una transición suave entre ambos pasos, manteniendo la alineación ya lograda y refinándola con un grado adicional de libertad (la escala).

### Componentes

- **Transform:** `Similarity3DTransform`, que incorpora rotación, traslación y escala uniforme.
- **Optimizer:** `RegularStepGradientDescentOptimizerv4`, igual que en el paso anterior.
- **Metric:** `MattesMutualInformationImageToImageMetricv4`, adecuada para comparar volúmenes multimodales o inter-sujetos.
- **Registration:** `ImageRegistrationMethodv4`.

### Parámetros usados

**Métrica**

- `SetNumberOfHistogramBins(50)`: Igual que en el paso anterior.
- `SetMetricSamplingPercentage`: Calculado como en el paso rígido, usando el número total de vóxeles y el número de muestras de 100,000.

**Optimizador**

- `SetLearningRate(0.2)` Valor heredado del paso anterior.
- `SetMinimumStepLength(0.01)`: Valor heredado del paso anterior.
- `SetNumberOfIterations(60)`: Puede que se requieran menos debido a que ya la rígida hizo unas modificaciones y no usó tantas.
- `SetRelaxationFactor(0.5)`: Por defecto.

### Iteraciones

La siguiente imagen muestra la evolución de la métrica durante el proceso de optimización similarity:

<img src="similarity_iterations.png" width="90%">


## Paso 3: Affine

En el tercer paso, refinamos aún más el registro permitiendo transformaciones lineales más generales mediante un modelo afín. Este paso introduce grados de libertad adicionales como cizallamiento (shear) y escalas no uniformes por eje, lo que lo hace ideal para capturar deformaciones globales más complejas entre imágenes.

### Transformación Afín

Utilizamos la clase `AffineTransform`, que permite:

- Rotación general.
- Escalado no uniforme.
- Shear.
- Traslación.

Partimos del resultado del paso anterior (`SimilarityTransform`). Para mantener la continuidad:

- Se copian el centro y la traslación directamente desde la transformación de similaridad.
- La matriz de rotación del `SimilarityTransform` se escala por su factor de escala para formar una nueva matriz afín.

Esto garantiza que la transformación afín inicial sea equivalente a la transformación de similaridad, pero con la capacidad de ser optimizada de forma más libre durante el registro.

### Componentes

- **Transform:** `AffineTransform`, 12 parámetros libres (9 de matriz + 3 de traslación).
- **Optimizer:** `RegularStepGradientDescentOptimizerv4`.
- **Metric:** `MattesMutualInformationImageToImageMetricv4`, consistente con los pasos anteriores.
- **Registration:** `ImageRegistrationMethodv4`.

### Parámetros usados

**Métrica**

- `SetNumberOfHistogramBins(50)`: igual que antes.
- `SetMetricSamplingPercentage`: misma lógica de muestreo proporcional al tamaño del volumen.

**Optimizador**

- `SetLearningRate(0.1)`: reducido frente a pasos anteriores para mantener estabilidad, dado que hay más parámetros a optimizar.
- `SetMinimumStepLength(0.001)`: sin cambios.
- `SetNumberOfIterations(60)`: sin cambios.
- `SetRelaxationFactor(0.5)`: sin cambios.

### Iteraciones

A continuación, se muestra la evolución de la métrica durante la optimización afín:

<img src="affine_iterations.png" width="90%">



## Paso 4: BSpline

### Parámetros

Primero se probó con la función `LBFGSBOptimizerv4` pero dio problemas con los parámetros `SetLineSearchAccuracy()` y `SetMaximumNumberOfIterations()`, ya que estos métodos no estaban disponibles en la versión de ITK instalada. El optimizador LBFGSB, aunque es teóricamente superior para problemas de alta dimensionalidad como el registro B-spline, presentó incompatibilidades de API entre diferentes versiones de ITK.

Posteriormente se intentó usar `ConjugateGradientLineSearchOptimizerv4` como alternativa, pero este optimizador no estaba disponible en el módulo ITK de la instalación actual, generando el error `AttributeError: module 'itk' has no attribute 'ConjugateGradientLineSearchOptimizerv4'`.

Finalmente se optó por usar `RegularStepGradientDescentOptimizerv4`, pero al usar un solo nivel (`SetNumberOfLevels(1)`, `SetShrinkFactorsPerLevel([1])`, `SetSmoothingSigmasPerLevel([0])`), sin importar los cambios de parámetros el tiempo de procesamiento era demasido alto que no se lograba completar ni siquiera una iteración. Incluso con tasas de aprendizaje del 1.0 para aumentar convergencia más rápida y usando relaxation factor de 0.5, el ordenador fallaba, sonaban los ventiladores y no logró encender por un tiempo debido a recalentamiento.

Finalmente, se probaron distintas opciones usando este optimizador para que diera tiempos coherentes y no saturara la máquina de computo:

- **Tamaño de la grilla**: Se pasó de (11, 11, 7) a (7, 7, 5) y (5, 5, 3) luego a para disminuir el uso excesivo de RAM.
- **`SetLearningRate(0.5)`**: Tasa de aprendizaje mucho mayor para permitir cambios significativos en los parámetros B-spline
- **`SetMinimumStepLength(0.05)`**: Mejor convergencia-
- **`SetNumberOfIterations(15)`**: Se dejan solo 15 porque en el segundo nivel cada iteración es muy costoso y si se usaban 200 llegaba a durar más de 6 horas sin terminar
- **`SetGradientMagnitudeTolerance(1e-6)`**: Tolerancia más estricta para asegurar convergencia real
- **Estrategia multi-resolución mejorada**: 3 niveles con factores de reducción `[2, 1]` y suavizado `[1, 0]`

Una grilla más fina requiere más parámetros de optimización (300 parámetros con la grilla actual vs. aprox 750 (7, 7, 5) vs. 2175 con la grilla original (11, 11, 7)), lo que hace que la convergencia sea más rápida y estable.

### Iteraciones

<img src="bspline_iterations.png" width="110%">

## Paso Final: Re-muestreo de PET usando la transformación de registro CT

### Parámetros y componentes

- (`LinearInterpolateImageFunction`) para preservar la calidad y suavidad de la imagen PET.
- **Valor de píxel por defecto**: 0 para las zonas fuera del dominio de la imagen móvil.
- **Transformación (`ct_transform`)**: Transformación compuesta resultante del registro CT (rígida, afín y/o B-spline) que se usará para alinear la PET.
- **Re-muestreador (`itk.ResampleImageFilter`)**: Filtrado de re-muestreo que aplica la transformación para interpolar la imagen PET en el espacio de referencia.

## Resultado CT

Se logró llegar a un valor de -0.436573 viendo mejoras en cada paso. El paso más costoso sin duda es el BSpline y tal vez para esta aplicación no se usó todo el poder porque los recursos de la máquina lo impidieron, pero tal vez usando una máquina con más recursos y dejando modificando los parámetros(disminuir tasa de aprendizaje, aumentar las iteraciones, dejar 3 niveles y una grilla más grande) se habría podido llegar muy buenos resultados parecidos a los que se obtienen con Slicer3D.

A continuación se muestra una comparación de cada paso visto en 3 vistas diferentes donde se combinan las imágenes de ambos sujetos. La imagen fija siempre es la que tiene un tono vidiris, fondo azul y zonas de alta intensidad como huesos en amarillo, la otra se dejó en escala de grises:

<img src="steps_merged.png" width="150%">

También se muestra el antes y despues de la iamgen PET:

<img src="pet_merged.png" width="100%">