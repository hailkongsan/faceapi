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

  // const labeledFaceDescriptors = await loadLabel()
  // const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)

  document.body.append('Loaded')

  input.addEventListener('change', async () => { 
    
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
