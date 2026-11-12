# Modelos oficiales usados, sin actualizacion automatica

Repositorio: https://github.com/opencv/opencv_zoo

Revision fijada: `48b10fa505aeb51fbd127fc60a5c660040e823b7`.
Los binarios se descargaron de media.githubusercontent.com del repositorio
oficial. SHA256 y tamano se contrastaron con los punteros Git LFS de esa
revision; el evaluador comprueba los hashes antes de cargar los modelos.
No utiliza un fork ni modelos convertidos localmente. No hay descarga en
los comandos de captura, prueba o auditoria.

| Modelo | Bytes | SHA256 | Licencia publicada en su directorio |
|---|---:|---|---|
| face_detection_yunet_2023mar.onnx | 232589 | 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4 | MIT, Shiqi Yu, 2020 |
| face_recognition_sface_2021dec.onnx | 38696353 | 0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79 | Apache-2.0 |

Origen exacto:

- [YuNet ONNX](https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)
- [SFace ONNX](https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/face_recognition_sface_2021dec.onnx)
- [Licencia YuNet](https://github.com/opencv/opencv_zoo/blob/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/LICENSE)
- [Licencia SFace](https://github.com/opencv/opencv_zoo/blob/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/LICENSE)

Las licencias y README completos estan incluidos junto a los modelos:
face_detection_yunet_LICENSE y face_recognition_sface_LICENSE. Estas son las
licencias que publica OpenCV Zoo para esos directorios; no se hace una
afirmacion adicional sobre los derechos de los conjuntos de entrenamiento.
El repositorio general usa Apache-2.0 y remite a las licencias por modelo.

El tutorial oficial documenta FaceDetectorYN/FaceRecognizerSF desde OpenCV
4.5.4 y coseno 0.363 para LFW:
https://docs.opencv.org/4.13.0/d0/dd4/tutorial_dnn_face.html

Ese umbral corresponde a un benchmark de verificacion y es PROVISIONAL en
esta prueba de identificacion abierta con varias referencias por persona.
No se presenta como calibracion local ni probabilidad de identidad.

Se eligio YuNet 2023mar estatico, con entrada letterbox exactamente 320x320,
para no depender del soporte de dimensiones dinamicas de motores diferentes.
Se deshace la escala de cajas y cinco landmarks antes de alignCrop sobre
el fotograma original. SFace alinea a 112x112 mediante su API oficial.
El modelo 2026may dinamico disponible en Zoo no se usa en este experimento.
