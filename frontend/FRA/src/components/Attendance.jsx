import React from 'react'
import { useNavigate } from 'react-router-dom'
import styled from 'styled-components'

function Attendance() {
  const navigate = useNavigate()
  return (
    <Style>
    <div className='content'>
      <h1>Get Ready for make today Attendance</h1>
      <h3>Have A Great Day</h3>
      <div className="note">
        Your Attendance Marking is ready, Dont Shake Look at Your Camera, After Fininshing click home Button to Return Home
      </div>
      <button onClick={()=>navigate('/')}>Home</button>
    </div>
    </Style>
  )
}
const Style = styled.div`
  button{
   width: 200px;
   padding: 5px;
   border-radius: 5px;
   background-color: #00BFFF;
  }
  color: aliceblue;
  .content{
    width: 100%;
    height: 400px;
    display: flex;
    flex-direction: column;
    justify-content: space-evenly;
    align-items: center;
  .note{
    color: red;
    width: 600px;
    border: 2px solid red;
    padding: 20px;
    border-radius: 10px;
  }
    
  }
`

export default Attendance