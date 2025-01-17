using Azure;
using Azure.AI.OpenAI;
using Azure.Identity; // Required for Passwordless auth

var endpoint = new Uri("https://qatarcentral.api.cognitive.microsoft.com/");
var credentials = new AzureKeyCredential("AzPH519Y43GTcK8rxSvE3mk987omBRblsr9DKcqCr65IoBoiXeLWJQQJ99ALACYeBjFXJ3w3AAAYACOGSvz6");
// var credentials = new DefaultAzureCredential(); // Use this line for Passwordless auth
var deploymentName = "whisper"; // Default deployment name, update with your own if necessary
var audioFilePath = "The Wind and the Sun - US English accent (TheFableCottage.com).mp3";

var openAIClient = new AzureOpenAIClient(endpoint, credentials);

var audioClient = openAIClient.GetAudioClient(deploymentName);

var result = await audioClient.TranscribeAudioAsync(audioFilePath);

Console.WriteLine("Transcribed text:");
foreach (var item in result.Value.Text)
{
    Console.Write(item);
}