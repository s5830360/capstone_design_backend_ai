package com.example.capstone_design.service;

import com.example.capstone_design.dto.AnalysisResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Service
public class AnalysisService {
    @Value("${app.analysis.url:http://localhost:8000/predict}")
    private String fastApiUrl;

    private final RestTemplate restTemplate = new RestTemplate();

    public AnalysisResponse analyzeFile(MultipartFile file) throws IOException {
        // FastAPI에 멀티파트 요청
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        // 파일 -> HttpEntity
        org.springframework.core.io.ByteArrayResource resource =
                new org.springframework.core.io.ByteArrayResource(file.getBytes()) {
                    @Override
                    public String getFilename() {
                        return file.getOriginalFilename();
                    }
                };

        MultiValueMap<String, Object> body = new org.springframework.util.LinkedMultiValueMap<>();
        body.add("file", resource);

        HttpEntity<MultiValueMap<String, Object>> requestEntity =
                new HttpEntity<>(body, headers);

        ResponseEntity<AnalysisResponse> response = restTemplate.exchange(
                fastApiUrl,
                HttpMethod.POST,
                requestEntity,
                AnalysisResponse.class
        );

        return response.getBody();
    }
}
