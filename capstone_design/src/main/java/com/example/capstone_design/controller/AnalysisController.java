package com.example.capstone_design.controller;

import com.example.capstone_design.dto.AnalysisResponse;
import com.example.capstone_design.service.AnalysisService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/analysis")
@RequiredArgsConstructor
public class AnalysisController {
    private final AnalysisService analysisService;

    @PostMapping("/temp")
    public ResponseEntity<AnalysisResponse> analyzeTemp(@RequestParam("file") MultipartFile file) throws Exception {
        AnalysisResponse result = analysisService.analyzeFile(file);
        return ResponseEntity.ok(result);
    }
}
