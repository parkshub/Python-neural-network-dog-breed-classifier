document.querySelector('button').addEventListener('click', collapse)

function collapse(){
    if (document.querySelector('button').innerText == 'show') {

        document.querySelector('.collapse').classList.remove('hidden')
        document.querySelector('button').innerText = 'hide'
    } else {
        document.querySelector('.collapse').classList.add('hidden')
        document.querySelector('button').innerText = 'show'
    }

}

const testcont = document.querySelector('#NOTHING')

const all = document.querySelectorAll('.friendImg')
Array.from(all).forEach(element => element.addEventListener('click', heartsFall))


function heartsFall(click) {//sounds like a band's name
    let friend = click.target
    let parent = friend.parentElement

    let classes = heartContainer.classList
    let class_array = Array.from(classes)

    let childNodes = Array.from(parent.childNodes)

    if (!(childNodes.includes(heartContainer))){
        parent.prepend(heartContainer)

        if ((class_array.includes('hideHearts'))){
            heartContainer.classList.remove('hideHearts')
        }
    } else {
        if (class_array.includes('hideHearts')){
            heartContainer.classList.remove('hideHearts')
        } else {
            heartContainer.classList.add('hideHearts')
        }
    }
}