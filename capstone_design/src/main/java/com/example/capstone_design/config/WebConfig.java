package com.example.capstone_design.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.*;

@Configuration
public class WebConfig implements WebMvcConfigurer{
    @Override
    public void addCorsMappings(CorsRegistry reg) {
        reg.addMapping("/api/**")
                .allowedOrigins("http://localhost:3000","http://localhost:5173","http://localhost:8081")
                .allowedMethods("GET","POST","DELETE","PUT","PATCH")
                .allowCredentials(true);
    }
}
