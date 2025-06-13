import { BrowserRouter, Routes, Route } from "react-router-dom";
import Registration from "./components/Registration";
import Layout from "./Components/Layout";
import Home from "./components/Home";
import Attendance from "./components/Attendance";
import SuccessPage from "./components/SuccessPage";
import Train from "./components/Train";
import AttendanceSuccess from "./components/AttendanceSuccess";

function App() {
  const message = "Face registered successfully"
  return (
    <>
    <BrowserRouter>
    <Routes>
      <Route>
        <Route path="/" element={<Layout/>}></Route>
        <Route index element={<Home/>}></Route>
        <Route path='/onboard' element={<Registration/>}></Route>
        <Route path='/Train' element={<Train/>}></Route>
        <Route path='/attendance' element={<Attendance/>}></Route>
        <Route path="/SuccessPage" element={<SuccessPage message={message}/>}></Route>
        <Route path="/AttendanceSuccess" element={<AttendanceSuccess/>}></Route>

      </Route>
    </Routes>
    </BrowserRouter>
    </>
  )
}

export default App