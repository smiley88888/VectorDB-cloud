<?php

$base_url = "http://24.199.123.128:8000";

// Create parameters array
$params = array(
    "id" => 11,
    "user_id" => 11,
    "text" => "Bad & Sanitär – ExpertenTesten.de"
);

// Create URL with parameters
$url = $base_url . "/insert?" . http_build_query($params);

// Initialize cURL session
$ch = curl_init();

// Set options
curl_setopt($ch, CURLOPT_URL, $url);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);

// Execute cURL session
$response = curl_exec($ch);

// Close cURL session
curl_close($ch);

// Print response
echo $response;

?>
