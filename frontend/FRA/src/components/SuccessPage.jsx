import React from 'react'
import styled from 'styled-components'
import check from '../assets/check-vmake.mp4'
import { useNavigate } from 'react-router-dom'

function SuccessPage({message}) {
    const navigate = useNavigate()
  return (
    <>
    <Style>
        <div className="success">
            {/* <img src="./src/assets/check.png" alt="" /> */}
            <video autoPlay muted>
                <source src={check} type='video/mp4'/>
            </video>
             <h1>Your {message} Registered Successfully</h1>
             <button onClick={()=>{navigate('/')}}>Home</button>
        </div>
    </Style>
    </>
  )
}
const Style = styled.div`
    div{
        width: 100%;
        height: 500px;
        display: flex;
        flex-direction: column;
        color: aliceblue;
        justify-content: space-evenly;
        align-items: center;
    }
    video{
        width: 300px;
        height: 300px;
    }
    button{
    color: #0A0F1C;
    padding: 5px;
    width: 200px;
    font-size: 15px;
    font-weight: bold;
    background-color: #00BFFF;
    border-radius: 5px;
  }
`
export default SuccessPage
