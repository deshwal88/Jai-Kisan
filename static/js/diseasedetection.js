var myObj=
{
    init:function(){
       var that=this;
       this.load_state();
       document.getElementById('state').addEventListener('change', function(){
           that.load_city(this.value);
    })
    
    },
    load_state:()=>{
        let xhr=new XMLHttpRequest();
        xhr.open('GET', 'https://raw.githubusercontent.com/bhanuc/indian-list/master/state-city.json',true);
        xhr.onload=()=>{
            let states=JSON.parse(xhr.responseText);
            Object.keys(states).forEach(value => {
                var op=document.createElement('option');
                op.innerText=value;
                op.setAttribute('value', value);
                document.getElementById('state').appendChild(op);
            });
         }
        xhr.send();
    },

     load_city:(statevalue)=>{
      document.getElementById('city').innerHTML='';
        let xhr=new XMLHttpRequest();

        xhr.open('GET', 'https://raw.githubusercontent.com/bhanuc/indian-list/master/state-city.json',true);
        xhr.onload=()=>{
           let states=JSON.parse(xhr.responseText);
           let arr=Object.keys(states[statevalue]);
           let obj=states[statevalue];
           let input=new Array();
           arr.forEach((i)=>{
            var op=document.createElement('option');
            op.innerText=obj[i];
            op.setAttribute('value', obj[i]);
           document.getElementById('city').appendChild(op);
          })
           }
         xhr.send();

     }

}

myObj.init();
