async function analyzeTweet() {

    const tweetInput =
        document.getElementById("tweet");

    const tweet =
        tweetInput.value.trim();


    const result =
        document.getElementById("result");

    const error =
        document.getElementById("error");

    const loading =
        document.getElementById("loading");


    // Clear previous results

    result.classList.add("hidden");

    error.classList.add("hidden");


    // Validate

    if (!tweet) {

        error.textContent =
            "Please enter a tweet.";

        error.classList.remove("hidden");

        return;
    }


    // Show loading

    loading.classList.remove("hidden");


    try {

        const response =
            await fetch(
                "/predict",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        text: tweet
                    })

                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Prediction failed."
            );

        }


        // Display result

        document.getElementById(
            "sentiment"
        ).textContent =
            data.sentiment;


        document.getElementById(
            "confidence"
        ).textContent =
            data.confidence + "%";


        result.classList.remove(
            "hidden"
        );


    } catch (errorObject) {

        error.textContent =
            errorObject.message;

        error.classList.remove(
            "hidden"
        );

    } finally {

        loading.classList.add(
            "hidden"
        );

    }

}