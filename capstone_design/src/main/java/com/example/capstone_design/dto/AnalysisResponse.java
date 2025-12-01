package com.example.capstone_design.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class AnalysisResponse {
    private String emotion;
    private Double confidence;
}
