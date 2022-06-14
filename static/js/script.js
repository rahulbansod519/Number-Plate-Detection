let toggled = false;
function toggle() {
    if (toggled == false) {
        toggled = true;
        let date_field = document.getElementById("filter");
        date_field.innerHTML = `<form class="form-container" action="/date-filter/video/" method="post">
        <div class="float-container">
            <div class="form-group float-child">
                <label for="date">Enter Date</label>
                <input type="date" class="form-control" id="date1" name="date1">
            </div>
            <div class="form-group float-child">
                <label for="exampleInputPassword1">Enter Date</label>
                <input type="date" class="form-control" id="date2" name="date2">
            </div>
            </div>
            <br>
                <button type="submit" class="btn btn-primary">Submit</button>
                <button onclick="toggle()" class="btn btn-primary">Change Filter</button>
            </form>
            </div >`
        return;
    }
    if (toggled == true) {
        toggled = false;
        let time_field = document.getElementById("filter");
        time_field.innerHTML = `<form class="form-container" action="/time-filter/video/" method="post">
            <div class="form-group">
            <label for="date">Enter Date</label>
            <input type="date" class="form-control" id="date" name="date_time" >
            </div>
            <div class="float-container">
            <div class="form-group float-child">
                <label for="time">From</label>
                <input type="time" class="form-control" id="time1" name="time1">
            </div>
            <div class="form-group float-child">
                <label for="time">To</label>
                <input type="time" class="form-control"  name="time2">
            </div>
            </div>
            <br>
                <button type="submit" class="btn btn-primary">Submit</button>
                <button onclick="toggle()" class="btn btn-primary">Change Filter</button>
            </form>
            </div >`
        return;
    }
}