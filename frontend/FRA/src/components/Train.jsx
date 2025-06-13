import axios from 'axios'
import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import styled from 'styled-components'

function Train() {
  const [message, setMessage] = useState("")
  const navigate = useNavigate()
  const train_model = async ()=>{
    setMessage("Please wait Your face is train to machine");
    try{
      const res = await axios.post('http://127.0.0.1:8000/train_model/');
      if(res.data.Status==="Success"){
        navigate('/SuccessPage')
      }
      else{
        alert("Worng if contition")
      }
    }
    catch(e){
      alert("Something went worng")
    }

  }
  return (
    <div>
      <Style>
        <button onClick={train_model}>Train Model</button>
        <h2>{message}</h2>
      </Style>
    </div>
  )
}
const Style = styled.div`
  button{
    color: #0A0F1C;
    padding: 5px;
    width: 200px;
    font-size: 15px;
    font-weight: bold;
    background-color: #00BFFF;
    border-radius: 5px;
  }
  color: white;
`

export default Train
