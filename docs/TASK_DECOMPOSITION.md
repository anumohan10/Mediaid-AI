# Task Decomposition for Complex Medical Queries

## Overview

MediAid AI now features **Advanced Task Decomposition** - an agentic AI capability that automatically breaks down complex medical queries into manageable sub-tasks for comprehensive analysis.

## How It Works

### 1. **Query Complexity Detection**
The system automatically identifies complex queries based on:
- Multiple medical conditions mentioned
- Complexity indicators (interactions, safety, pregnancy, etc.)
- Query length and structure
- Safety considerations

### 2. **Automatic Decomposition**
Complex queries are broken down into:
- **Main Topic**: Central medical question
- **Conditions**: Individual medical conditions to research
- **Sub-questions**: Specific aspects to address
- **Safety Considerations**: Drug interactions, contraindications
- **Complexity Level**: Low/Medium/High classification

### 3. **Multi-faceted Research**
The system conducts separate searches for:
- Each medical condition (symptoms, treatment, management)
- Safety considerations and drug interactions
- Specific sub-questions from the decomposition

### 4. **Intelligent Synthesis**
All research results are combined into a comprehensive response with:
- **Overview**: Summary of the main question
- **Condition Analysis**: Detailed information for each condition
- **Safety Considerations**: Important warnings and interactions
- **Specific Recommendations**: Answers to sub-questions
- **Action Plan**: Clear next steps
- **Medical Disclaimer**: Professional consultation reminder

## Example Transformations

### Simple Query (Standard Processing)
**Input:** "What is diabetes?"
**Processing:** Single search + AI summary
**Output:** General diabetes information

### Complex Query (Task Decomposition)
**Input:** "I have diabetes and high blood pressure and I'm pregnant. What medications are safe?"

**Decomposition:**
```json
{
  "main_topic": "Safe medications for pregnant woman with diabetes and hypertension",
  "conditions": ["diabetes", "hypertension", "pregnancy"],
  "sub_questions": [
    "What diabetes medications are safe during pregnancy?",
    "What blood pressure medications are safe during pregnancy?",
    "Are there any drug interactions between diabetes and BP medications?"
  ],
  "safety_considerations": [
    "pregnancy medication safety",
    "diabetes-hypertension drug interactions",
    "maternal-fetal medication effects"
  ],
  "complexity_level": "high"
}
```

**Multi-faceted Research:**
1. **Diabetes Research**: Management during pregnancy, safe medications
2. **Hypertension Research**: Pregnancy-safe blood pressure treatments
3. **Pregnancy Safety**: Medication categories, contraindications
4. **Drug Interactions**: Diabetes-hypertension medication combinations

**Synthesized Response:**
- Comprehensive analysis covering all conditions
- Specific medication recommendations
- Safety warnings and contraindications
- Clear action plan for medical consultation

## Benefits of Task Decomposition

### 🎯 **More Accurate Responses**
- Addresses all aspects of complex questions
- Reduces chance of missing important information
- Provides comprehensive coverage

### 🧠 **Intelligent Analysis**
- Automatically identifies key research areas
- Prioritizes safety considerations
- Structures information logically

### ⚡ **Efficient Processing**
- Parallel research across multiple topics
- Focused searches for each sub-task
- Optimized resource utilization

### 🔍 **Better Source Attribution**
- Clear categorization of research sources
- Condition-specific references
- Safety-focused documentation

## When Task Decomposition Activates

### **Automatic Triggers:**
- Multiple medical conditions mentioned
- Drug interaction queries
- Pregnancy/elderly considerations
- Queries longer than 15 words
- Safety-related keywords

### **Example Trigger Phrases:**
- "I have [condition1] and [condition2]..."
- "What medications are safe during..."
- "Drug interactions between..."
- "Elderly patient with multiple conditions..."
- "Side effects of taking both..."

## Technical Implementation

### **Core Functions:**
1. `is_complex_query()` - Detects query complexity
2. `decompose_medical_query()` - Breaks down queries using AI
3. `execute_complex_search()` - Conducts multi-faceted research
4. `synthesize_complex_response()` - Combines results intelligently

### **AI Models Used:**
- **GPT-3.5-turbo** for query decomposition
- **OpenAI Embeddings** for semantic search
- **FAISS** for fast document retrieval
- **LlamaIndex** for intelligent querying

## User Experience

### **Visual Indicators:**
- 🧠 "Complex Query Detected" notification
- 🔍 Query analysis breakdown shown to user
- 📊 Research progress indicators
- 📚 Categorized source display

### **Response Structure:**
1. **Query Analysis** (expandable section)
2. **Comprehensive Response** (main content)
3. **Research Sources by Category** (expandable)
4. **Action Plan** (clear next steps)

## Future Enhancements

### **Planned Features:**
- **Learning from History**: Improve decomposition based on user interactions
- **Dynamic Tool Selection**: Choose best search methods per sub-task
- **Risk Assessment**: Automatic identification of high-risk combinations
- **Personalized Decomposition**: Adapt to user's medical history
- **Multi-language Support**: Decomposition in multiple languages

### **Advanced Capabilities:**
- **Temporal Reasoning**: Consider timing of medications
- **Dosage Calculations**: Factor in patient-specific parameters
- **Alternative Treatments**: Suggest backup options
- **Research Paper Integration**: Include latest medical literature

## Best Practices

### **For Users:**
- Be specific about all medical conditions
- Mention current medications
- Include relevant demographic info (age, pregnancy, etc.)
- Ask follow-up questions for clarification

### **For Developers:**
- Monitor decomposition accuracy
- Regularly update complexity detection patterns
- Validate medical information quality
- Maintain comprehensive error handling

## Conclusion

Task Decomposition transforms MediAid AI from a simple search tool into an intelligent medical research assistant that can handle complex, multi-faceted medical queries with the thoroughness and systematic approach of a medical professional.

This agentic AI capability ensures that users receive comprehensive, well-researched responses that address all aspects of their complex medical questions while maintaining the highest standards of safety and accuracy.
