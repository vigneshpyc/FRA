import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import styled from 'styled-components'
import axios from 'axios'
function Registration() {
  const navigate = useNavigate()
  const [emp_data, setEmpData] = useState(
    {
      emp_nama : ""
    }
  )
  const emp_get_data = (e)=>{
    setEmpData({...emp_data,[e.target.name]:e.target.value})
  }

  const handlesubmit = async (e)=>{
    e.preventDefault();
    const name_value = new FormData()
    name_value.append('name',emp_data.name)
    try{
      const respose = await axios.post("http://127.0.0.1:8000/name_receive/",name_value,{headers:{'Content-Type':'multipart/form-data'}})
      if(respose.data.Status==="Success"){
        navigate('/Train')
      }
    }
    catch(e){
      alert("Something went worng")
    }
  }

  return (
    <>
      <Style>
        
        <button onClick={()=>navigate('/')}>Home</button>
      <div className="heading">
        <h2>OnBoard</h2>
      </div>
      <div className="form">
    
        <form action="#" onSubmit={handlesubmit}>
          <div className="inputs">
            <label htmlFor="emp_name">Enter Employee name </label>
            <input type="text" name='name' value={emp_data.name} onChange={emp_get_data} /><br />
            <label htmlFor="age">Age</label>
            <input type="numbers" /><br />
            <label htmlFor="designation">Designation</label>
            <input type="text" /><br />
          </div>
          <button type='submit'>submit</button>
        </form>
      </div>
      </Style>
    </>
  )
}
const Style = styled.div`
color: #00BFFF;
*{
  z-index: 1;
}
  .heading h2{
    width: 150px;
    background-color: #00BFFF;
    color: #0A0F1C;
    display: inline-block;
    padding: 10px;
    border-radius:10px;
    text-align: center;
    font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
  }
  .heading{
    height: 100px;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  form{
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    align-items: center;
  }
  .inputs{
    display: flex;
    flex-direction: column;
  }
  
  input{
    background-color: #D9D9D9;
    border: none;
    width: 250px;
    font-size: 15px;
    border-radius: 5px;
    outline-color: #00BFFF;
    padding: 5px;
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

export default Registration
