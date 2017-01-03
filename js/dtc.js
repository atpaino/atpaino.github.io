$(document).ready(function() {

  var onSuccess = function(data) {
      $("#dtc-corrected").html(data["output"]);
  };

  $("#dtc-form").submit(function(e){
        var input = $("#text-to-correct").val();

        var url = "https://2hbqifxaxh.execute-api.us-east-1.amazonaws.com/test/correct-text?text=" + input;
        

        $.ajax(url,{
                dataType: "json",
                success: onSuccess});
        return false;
  });
});
