const input = document.getElementById('imageUpload')

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.faceExpressionNet.loadFromUri('/models'),
  faceapi.nets.ageGenderNet.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
]).then(start)

async function start() {
  const container = document.createElement('div')
  container.style.position = 'relative'
  document.body.append(container)
  
  const labeledFaceDescriptors = await loadLabel()
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)

  document.body.append('Loaded')

  let image
  let canvas

  input.addEventListener('change', async () => {
    image = await faceapi.bufferToImage(input.files[0])

    canvas = faceapi.createCanvasFromMedia(image)
    
    const detections =  await faceapi.detectAllFaces(image)
      .withFaceLandmarks()
      .withFaceExpressions()
      .withFaceDescriptors()
      .withAgeAndGender()

      faceapi.draw.drawFaceLandmarks(canvas, detections)
      
      detections.forEach((detection) => {
        const { age, gender, genderProbability } = detection
        let expression = Object.entries(detection.expressions).reduce((a, b) => {
          return a[1] > b[1] ? a : b
        })
        
        new faceapi.draw.DrawTextField(
          [
            `Age: ${Math.round(age)}`,
            `Gender: ${gender} (${Number.parseFloat(genderProbability).toFixed(2)})`,
            `Expression: ${expression[0]} (${Number.parseFloat(expression[1].toFixed(2))})`
          ],
          detection.detection.box.bottomLeft
          ).draw(canvas)
      })

      const results = detections.map(d => faceMatcher.findBestMatch(d.descriptor))

      results.forEach((result, i) => {
        new faceapi.draw.DrawBox(detections[i].detection.box, {
          label: result.toString()
        }).draw(canvas)
      })

      document.body.append(canvas)
  })
}


function loadLabel()
{
  const labels = ['Black Widow', 'Captain America', 'Captain Marvel', 'Hawkeye', 'Jim Rhodes', 'Thor', 'Tony Stark']

  return Promise.all(
    labels.map(async lable => {
      const decriptions = []
      for (let i = 1; i < 2; i++) {
        const img = await faceapi.fetchImage(`./labeled_images/${lable}/${i}.jpg`)
        const detection = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()

        decriptions.push(detection.descriptor)
      }

      return new faceapi.LabeledFaceDescriptors(lable, decriptions)
    })
  )
}
