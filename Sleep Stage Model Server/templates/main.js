currentSlideId=0;
sliderElement=document.getElementById('slider');
totalSlides=sliderElement.childElementCount;
console.log(totalSlides);

function showSlide()
{
    console.log('slide shown');
    slides=document.getElementById('slider').getElementsByTagName('li')
    for (let index = 0; index < totalSlides; index++) {
        const element = slides[index];
        if (currentSlideId===index)
        {
            element.classList.remove('hidden')
        }
        else
        {
            element.classList.add('hidden')
        }
    }
}

function next()
{
    if (totalSlides > currentSlideId)
    {
        currentSlideId=(currentSlideId+1)%totalSlides;
        console.log(currentSlideId);
        showSlide();
    }
    console.log('next');
}


function prev()
{
    if (currentSlideId > 0)
    {
        currentSlideId--;
        console.log(currentSlideId);
        showSlide();
    }
    console.log('prev');
}


function autoplay()
{
    next();
    console.log('called');
    if (totalSlides > currentSlideId)
    {
        setTimeout(() => {
            autoplay();
        }, 3000);
    }
}

setTimeout(() => {
    autoplay();
}, 3000);


