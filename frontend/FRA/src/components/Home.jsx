import React from 'react'
import { useNavigate } from 'react-router-dom'
import './home.css'
import styled from 'styled-components'
import axios from 'axios'
import attendance from '../assets/attendance.jpg'
function Home() {
  const navigate = useNavigate()
  const navigation = async ()=>{
    navigate('/attendance')
    try{
      const res = await axios.post("http://127.0.0.1:8000/attendance/")
      if(res.data.Status === "Success"){
        navigate('/AttendanceSuccess')
      }
    }
    catch(e){
      alert("Something went wrong")
    }
    
  }
  return (
    <div>
      <Style>
        <nav>

        </nav>
        <div className="content">
          <h1>Smart.Fast.Contactless</h1>
          <p>Mark your attendance with just a look. Register your face once — check in effortlessly every time.
No cards. No clicks. Just you</p>
        </div>
        <div className="btn">
        <div className="onboard">
          <div className="pic">
            <img src='../src/assets/face.jpg' width={100} height={100} alt="some technical error" />
          </div>
          <button onClick={()=>navigate('/onboard')}>Onboard</button>
        </div>
        <div className="attendance">
          <div className="pic">
            <img src='../src/assets/attendance.jpg' width={100} height={100} alt="" />
          </div>
          <button onClick={navigation}>Attendance</button>
        </div>
        </div>
      </Style>
    </div>
  )
}
const Style = styled.div`
  button{
    color: #0A0F1C;
    padding: 10px;
    background-color: #C4D9FF;
    border-radius:5px;
    margin:1rem;
  }
  .btn{
    display: flex;
    justify-content:space-evenly;
    align-items: center;
  }
  .content{
    display: flex;
    align-items: center;
    flex-direction: column;
    justify-content: center;
    color: #1E90FF;
    font-family: 'Courier New', Courier, monospace;
    padding: 50px;
  }
  .content p{
    color: #C4D9FF;
    font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
    width: 400px;
    text-align: center;
  }
  .onboard,.attendance{
    background-color: #00BFFF;
    width: 200px;
    height:300px;
    margin: 10px;
    padding: 0;
    display: flex;
    flex-direction: column;
    justify-content: space-evenly;
    align-items: center;
    border-radius: 10px;
  }
  .pic{
    width:100px;
    height: 100px;
    border-radius: 50%;
    background-color: #C4D9FF;
    background-repeat: no-repeat;
    background-size: cover;
  }
  .pic img{
    border-radius: 50%;
  }
  nav{
    width: 100%;
    height: 50px;
    background-color: #1E90FF;
  }
`

export default Home
