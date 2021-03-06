var ajaxRequestCount = 0;
var nextButtonPressed = false;

var resultHandlerManual = function(result) {
    // Do NOT over-write the sub-price fields as this causes unwanted behavior if there is network latency

    // Updating via JS instead to avoid race conditions
    //$("#id_ask_total").val(result.ask_total); // updating via js first
    //$("#id_ask_stdev").val(result.ask_stdev);
    ajaxRequestCount--;
    console.log(ajaxRequestCount);
};
var resultHandlerAuto = function(result) {
    // Over-writing user-input to enforce consistency.  The only reason this should ever be different
    //  is if the user tried to enter a decimal-point

    $.each(result.pricedims, function(i, pd){
        $("#dim_" + (i+1)).val(pd);
    });

    $("#id_ask_total").val(result.ask_total);
    $("#id_ask_stdev").val(result.ask_stdev);
};

$(document).ready(function() {
    // when this is called from the instructions page, the example flag should be set to True, which should prevent
    // the server from adding a row to the Ask database.

    // in utils.js
    setup_csrf();

    $("#id_ask_total").focus(function(){
        // If a user brings the ask_total field into focus, then the dim fields
        //  get reset to empty.  This prevents the user from entering values
        //  into sub-prices, and then entering a total price that is unequal to
        //  the sum of the already-entered sub-prices.
        $(".pricedim").val("");
    });

    $(".pricedim").change(function () {
        // When player manually changes a field, we want to send the whole list of fields back as a list

        var pricedims = [];
        var sum = 0;
        $(".pricedim").each(function (i, input) {
            var val = parseInt($(input).val()) ? parseInt($(input).val()) : 0;
            pricedims.push(val);
            sum += val;
        });
        var avg = sum/pricedims.length;
        var std = 0;
        $(pricedims).each(function (i, val) {
            std += Math.pow(val - avg, 2) / pricedims.length;
        });
        std = Math.pow(std, 0.5) 
;
        // Set totals variables via js
        $("#id_ask_total").val(sum);
        $("#id_ask_stdev").val(std);

        var data = get_metadata($("#distribute"));
        data.pricedims = pricedims.toString();

        console.log(pricedims.toString())
        console.log(data.pricedims.toString())
        console.log("Changed a price dimension")

        $.ajax({
            type: "POST",
            url: $("#distribute").attr("data-manual-url"),
            data: data,
            dataType: "json",
            beforeSend: function(){
                ajaxRequestCount++;
                console.log(ajaxRequestCount);
            },
            success: resultHandlerManual
        });
    });

    // When a player clicks the automatic distribute button, we ask the server for the list of price dims
    $("#distribute").click(function () {
        // #Distribute field validation.  Don't send if number not in range.
        var myForm = $("form");
        var myVal = $("#id_ask_total").val();
        if (!myForm[0].checkValidity() && myVal > 800 || myVal < 0 || myVal == "") {
            // Leveraging built-in styling for "distribute" validation
            // Validation technique taken from
            //   http://stackoverflow.com/questions/10092580/stop-form-from-submitting-using-jquery#10092636
            // If the form is invalid, submit it. The form won't actually submit;
            //   this will just cause the browser to display the native HTML5 error messages.
            $(".seller_widget input[type='submit']")[0].click();
            return;
        }

        var data = get_metadata($(this));
        data.ask_total = $("#id_ask_total").val();
        data.numdims = $("input.pricedim").length;

        $.ajax({
            type: "POST",
            url: $("#distribute").attr("data-auto-url"),
            data: data,
            dataType: "json",
            success: resultHandlerAuto
        });
    });

    $('.a-btn').click(function(){
        $(".a-btn").prop("disabled", true);
        $(".pricedim").prop("disabled", true);
        nextButtonPressed = true;
        if (ajaxRequestCount == 0){
            console.log("Now proceeding to next page");
            $(".a-btn").prop("disabled", false);
            $(".pricedim").prop("disabled", false);
            $("nav input[type='submit']").click();
        }
    });
})

$(document).ajaxStop(function(){
    console.log("Finished AJAX requests")
    if(ajaxRequestCount == 0 && nextButtonPressed){
        console.log("Now proceeding to next page");
        nextButtonPressed = false;
        $("nav input[type='submit']").click();
        $(".a-btn").prop("disabled", false);
        $(".pricedim").prop("disabled", false);
    }
})