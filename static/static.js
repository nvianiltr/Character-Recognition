(function() {
	var canvas = document.querySelector("#canvas");
	var context = canvas.getContext("2d");
	canvas.width = 660;
	canvas.height = 280;

	var loc = {x:0, y:0};
	var prev = {x:0, y:0};
	context.fillStyle = "white";
	context.fillRect(0, 0, canvas.width, canvas.height);
	context.color = "black";
	context.lineWidth = 7;
    context.lineJoin = context.lineCap = 'round';

	debug();

	$("#pen").on("click", function() {
		$("#eraser").removeAttr("disabled");
		$("#pen").attr("disabled", "disabled");
    	context.color="black";
    });

	$("#eraser").on("click", function() {
		$("#pen").removeAttr("disabled");
		$("#eraser").attr("disabled", "disabled");
    	context.color="white";
    });

	canvas.addEventListener("mousemove", function(e) {
		prev.x = loc.x;
		prev.y = loc.y;

		loc.x = e.pageX - this.offsetLeft-15;
		loc.y = e.pageY - this.offsetTop-15;
	}, false);
	canvas.addEventListener("mousedown", function(e) {
		canvas.addEventListener("mousemove", onPaint, false);
	}, false);
	canvas.addEventListener("mouseup", function() {
		canvas.removeEventListener("mousemove", onPaint, false);
	}, false);

	var onPaint = function() {
		context.lineWidth = context.lineWidth;
		context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;

		context.beginPath();
		context.moveTo(prev.x, prev.y);
		context.lineTo(loc.x,loc.y );
		context.closePath();
		context.stroke();
	};

	function debug() {
		$("#clear_button").on("click", function() {
			context.clearRect( 0, 0, 280, 280 );
			context.fillStyle="white";
			context.fillRect(0,0,canvas.width,canvas.height);
		});
	}
}());
