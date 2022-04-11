# AnimeGAN API

Author: Aniket Thomas (AniTho)

# Overview

This consists of an API built using fastapi to use AnimeGan 

# Usage

Run the main file and call the api from your application through the path (\<ip-address of api>/api/trans/) and pass the following inputs.

Input:<br> 
&emsp;- Image Byte: Passed as a file upload<br>
&emsp;&emsp;Passes the Image on which AnimeGan Model has to be used <br>
<br>
&emsp;- model_name: Passed as a json request <br>
&emsp;&emsp;Specifies the weights to be used for inference. <br>
&emsp;&emsp;Supports 'celeba', 'facepaintv1', 'facepaitv2', 'hayao', 'paprika', 'shinkai'
<br>
<br>

The api returns a byte object of the transformed image as a response.

Returns:<br>
Returns a byte object of the transformed image after applying the animegan model

<strong>Note</strong>: A sample Use case is shown in test.ipynb notebook.