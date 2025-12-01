package com.example.capstone_design.repository;

import com.example.capstone_design.entity.Recording;
import org.springframework.data.jpa.repository.JpaRepository;

public interface RecordingRepository extends JpaRepository<Recording, Long>{
}
